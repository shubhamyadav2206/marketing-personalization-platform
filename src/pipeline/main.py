import os
import sys
import logging
import time
from pyspark.sql import SparkSession
from pyspark import SparkConf
from typing import Optional

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import pipeline components
from src.pipeline.ingestion import ingest_data
from src.pipeline.embeddings import generate_embeddings_spark
from src.pipeline.vector_store import store_vectors
from src.pipeline.graph_store import build_graph
from src.pipeline.analytics import aggregate_metrics
from src.pipeline.document_store import store_to_mongodb
from src.pipeline.schema_validator import (
    validate_schema, 
    validate_embeddings, 
    detect_anomalies,
    DataLineageTracker,
    calculate_dataframe_hash
)
from src.pipeline.analytics_db import AnalyticsDB
from src.pipeline.monitoring import get_pipeline_monitor
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

def setup_spark() -> SparkSession:
    """Initialize and configure a Spark session."""
    conf = SparkConf()
    
    # Set common configurations
    conf.setAppName("MarketingPersonalizationPipeline")
    conf.set("spark.sql.shuffle.partitions", "4")
    conf.set("spark.driver.memory", "4g")
    conf.set("spark.executor.memory", "4g")
    
    # Add PostgreSQL JDBC driver (update path as needed)
    spark_jars = os.getenv("SPARK_JARS")
    if spark_jars:
        conf.set("spark.jars", spark_jars)
    
    # Create and return the Spark session
    spark = SparkSession.builder \
        .config(conf=conf) \
        .getOrCreate()
    
    logger.info("Spark session initialized")
    return spark

def run_pipeline(input_path: Optional[str] = None):
    """
    Execute the complete marketing personalization pipeline.
    
    Args:
        input_path: Optional path to input data file. If None, uses default.
    """
    logger.info("Starting marketing personalization pipeline")
    start_time = time.time()
    
    # Initialize Spark
    spark = setup_spark()
    
    # Initialize monitoring and lineage tracking
    monitor = get_pipeline_monitor()
    lineage_tracker = DataLineageTracker()
    run_id = str(uuid.uuid4())
    monitor.start_run(run_id, input_path)
    
    try:
        # 1. Data Ingestion
        logger.info("=== Starting Data Ingestion ===")
        ingestion_start = time.time()
        df_raw = ingest_data(spark, file_path=input_path)
        input_count = df_raw.count()
        ingestion_duration = time.time() - ingestion_start
        logger.info(f"Ingested {input_count} records in {ingestion_duration:.2f}s")
        
        # Schema validation
        logger.info("=== Validating Schema ===")
        is_valid, errors = validate_schema(df_raw)
        if not is_valid:
            logger.error(f"Schema validation failed: {errors}")
            monitor.end_run("failed", f"Schema validation failed: {errors}")
            raise ValueError(f"Schema validation failed: {errors}")
        
        # Anomaly detection
        logger.info("=== Detecting Anomalies ===")
        anomalies = detect_anomalies(df_raw)
        logger.info(f"Anomalies detected: {anomalies}")
        
        input_hash = calculate_dataframe_hash(df_raw)
        lineage_tracker.record_step(
            "ingestion",
            input_count,
            input_count,
            input_hash=input_hash,
            metadata={"anomalies": anomalies, "duration_seconds": ingestion_duration}
        )
        monitor.record_step("ingestion", ingestion_duration, input_count, input_count, 
                           errors=errors if errors else None, anomalies=anomalies)
        
        # 2. Store to MongoDB (text + metadata)
        logger.info("\n=== Storing to MongoDB ===")
        mongo_start = time.time()
        mongo_result = store_to_mongodb(df_raw)
        mongo_duration = time.time() - mongo_start
        mongo_output = mongo_result['inserted'] + mongo_result['updated']
        logger.info(f"Stored {mongo_result['total']} documents to MongoDB in {mongo_duration:.2f}s")
        lineage_tracker.record_step(
            "mongodb_storage",
            input_count,
            mongo_output,
            metadata={"inserted": mongo_result['inserted'], "updated": mongo_result['updated'], 
                     "errors": mongo_result['errors'], "duration_seconds": mongo_duration}
        )
        monitor.record_step("mongodb_storage", mongo_duration, input_count, mongo_output,
                           errors=[f"errors: {mongo_result['errors']}"] if mongo_result['errors'] > 0 else None)
        
        # 3. Generate Embeddings
        logger.info("\n=== Generating Embeddings ===")
        embedding_start = time.time()
        df_with_embeddings = generate_embeddings_spark(spark, df_raw)
        embedding_count = df_with_embeddings.count()
        embedding_duration = time.time() - embedding_start
        logger.info(f"Generated embeddings for {embedding_count} records in {embedding_duration:.2f}s")
        
        # Validate embeddings
        logger.info("=== Validating Embeddings ===")
        emb_valid, emb_errors = validate_embeddings(df_with_embeddings)
        if not emb_valid:
            logger.error(f"Embedding validation failed: {emb_errors}")
            monitor.end_run("failed", f"Embedding validation failed: {emb_errors}")
            raise ValueError(f"Embedding validation failed: {emb_errors}")
        
        output_hash = calculate_dataframe_hash(df_with_embeddings)
        lineage_tracker.record_step(
            "embedding_generation",
            input_count,
            embedding_count,
            input_hash=input_hash,
            output_hash=output_hash,
            metadata={"duration_seconds": embedding_duration, "embedding_errors": emb_errors}
        )
        monitor.record_step("embedding_generation", embedding_duration, input_count, embedding_count,
                           errors=emb_errors if emb_errors else None)
        
        # 4. Store in Vector Database (Milvus)
        logger.info("\n=== Storing in Vector Database ===")
        milvus_start = time.time()
        vector_store = store_vectors(df_with_embeddings)
        milvus_duration = time.time() - milvus_start
        logger.info(f"Successfully stored vectors in Milvus in {milvus_duration:.2f}s")
        lineage_tracker.record_step(
            "milvus_storage",
            embedding_count,
            embedding_count,
            metadata={"duration_seconds": milvus_duration}
        )
        monitor.record_step("milvus_storage", milvus_duration, embedding_count, embedding_count)
        
        # 5. Build Knowledge Graph (Neo4j)
        logger.info("\n=== Building Knowledge Graph ===")
        neo4j_start = time.time()
        build_graph(df_with_embeddings)
        neo4j_duration = time.time() - neo4j_start
        logger.info(f"Successfully built knowledge graph in Neo4j in {neo4j_duration:.2f}s")
        lineage_tracker.record_step(
            "neo4j_graph_build",
            embedding_count,
            embedding_count,
            metadata={"duration_seconds": neo4j_duration}
        )
        monitor.record_step("neo4j_graph_build", neo4j_duration, embedding_count, embedding_count)
        
        # 6. Run Analytics (PostgreSQL or SQLite)
        logger.info("\n=== Running Analytics ===")
        analytics_start = time.time()
        
        # Determine which DB to use (default: sqlite for simplicity, postgresql for production)
        db_type = os.getenv('ANALYTICS_DB_TYPE', 'sqlite').lower()
        if db_type == 'sqlite':
            db_config = {
                'type': 'sqlite',
                'path': os.getenv('ANALYTICS_DB_PATH', 'analytics.db')
            }
        else:
            db_config = {
                'type': 'postgresql',
                'host': os.getenv('POSTGRES_HOST', 'localhost'),
                'port': os.getenv('POSTGRES_PORT', '5432'),
                'database': os.getenv('POSTGRES_DB', 'analytics'),
                'user': os.getenv('POSTGRES_USER', 'user'),
                'password': os.getenv('POSTGRES_PASSWORD', 'password')
            }
        
        # Use the new AnalyticsDB for consistency
        analytics_db = AnalyticsDB(db_config)
        analytics_db.create_tables()
        
        # Still use aggregate_metrics for now, but pass the db config
        aggregate_metrics(spark, df_with_embeddings, db_config)
        
        # Update campaign engagement frequency
        analytics_db.update_campaign_engagement_frequency(df_with_embeddings)
        
        analytics_duration = time.time() - analytics_start
        logger.info(f"Analytics pipeline completed successfully in {analytics_duration:.2f}s")
        lineage_tracker.record_step(
            "analytics_aggregation",
            embedding_count,
            embedding_count,
            metadata={"db_type": db_type, "duration_seconds": analytics_duration}
        )
        monitor.record_step("analytics_aggregation", analytics_duration, embedding_count, embedding_count)
        
        # Save lineage tracking
        lineage_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data",
            "reports",
            f"lineage_{lineage_tracker.pipeline_run_id.replace(':', '-')}.json"
        )
        lineage_tracker.save_lineage(lineage_path)
        
        # Save monitoring metrics
        monitor.end_run("completed")
        monitor.save_metrics()
        
        # Detect latency anomalies
        latency_anomalies = monitor.detect_latency_anomalies()
        if latency_anomalies:
            logger.warning(f"Detected {len(latency_anomalies)} latency anomalies: {latency_anomalies}")
        
        # Log completion
        duration = time.time() - start_time
        logger.info(f"\n=== Pipeline completed in {duration:.2f} seconds ===")
        logger.info(f"Pipeline metrics: ingestion={ingestion_duration:.2f}s, "
                   f"mongodb={mongo_duration:.2f}s, embedding={embedding_duration:.2f}s, "
                   f"milvus={milvus_duration:.2f}s, neo4j={neo4j_duration:.2f}s, "
                   f"analytics={analytics_duration:.2f}s")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
        monitor.end_run("failed", str(e))
        monitor.save_metrics()
        lineage_tracker.record_step(
            "pipeline_failure",
            0,
            0,
            metadata={"error": str(e)}
        )
        raise
    
    finally:
        # Clean up resources
        spark.stop()
        logger.info("Spark session stopped")

if __name__ == "__main__":
    import argparse
    import time
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Marketing Personalization Pipeline")
    parser.add_argument(
        "--input", 
        type=str, 
        default=None,
        help="Path to input data file (default: use built-in sample data)"
    )
    
    args = parser.parse_args()
    
    # Run the pipeline
    run_pipeline(input_path=args.input)
