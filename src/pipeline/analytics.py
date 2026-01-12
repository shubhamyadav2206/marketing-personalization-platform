from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, TimestampType, DateType
import logging
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
from sqlalchemy import create_engine, text
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_postgres_connection(spark: SparkSession, db_config: Dict):
    """Set up PostgreSQL connection properties."""
    return (spark.read
            .format("jdbc")
            .option("url", f"jdbc:postgresql://{db_config['host']}:{db_config['port']}/{db_config['database']}")
            .option("user", db_config['user'])
            .option("password", db_config['password'])
            .option("driver", "org.postgresql.Driver"))

def create_analytics_tables(spark: SparkSession, db_config: Dict):
    """Create necessary tables in PostgreSQL for analytics."""
    try:
        engine = create_engine(
            f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )

        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS user_engagement (
                    user_id TEXT,
                    message_count BIGINT,
                    engagement_score TEXT,
                    last_active TIMESTAMP,
                    preferred_campaigns TEXT
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS campaign_performance (
                    campaign_id TEXT,
                    total_messages BIGINT,
                    unique_users BIGINT,
                    avg_sentiment DOUBLE PRECISION,
                    last_updated TIMESTAMP
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS user_activity_daily (
                    activity_date DATE,
                    user_id TEXT,
                    message_count INT,
                    campaigns_interacted TEXT
                )
            """))

        logger.info("Analytics tables created/validated in PostgreSQL")
        
    except Exception as e:
        logger.error(f"Error creating analytics tables: {e}")
        raise

def calculate_user_engagement(df: DataFrame) -> DataFrame:
    """Calculate user engagement metrics."""
    logger.info("Calculating user engagement metrics...")
    
    # Count messages per user
    user_engagement = df.groupBy("userid") \
        .agg(
            F.count("*").alias("message_count"),
            F.collect_list("campaign").alias("campaigns"),
            F.max("timestamp").alias("last_active")
        )
    
    # Calculate engagement score
    user_engagement = user_engagement.withColumn(
        "engagement_score",
        F.when(F.col("message_count") > 10, "high")
         .when(F.col("message_count") > 5, "medium")
         .otherwise("low")
    )
    
    # Get top 3 campaigns per user
    user_engagement = user_engagement.withColumn(
        "preferred_campaigns",
        F.slice(F.array_distinct(F.col("campaigns")), 1, 3)
    )
    
    # Select final columns
    user_engagement = user_engagement.select(
        F.col("userid").alias("user_id"),
        "message_count",
        "engagement_score",
        "last_active",
        "preferred_campaigns"
    )
    
    return user_engagement

def calculate_campaign_performance(df: DataFrame) -> DataFrame:
    """Calculate campaign performance metrics."""
    logger.info("Calculating campaign performance metrics...")
    
    # Basic campaign metrics
    campaign_performance = df.groupBy("campaign") \
        .agg(
            F.count("*").alias("total_messages"),
            F.countDistinct("userid").alias("unique_users"),
            F.max("timestamp").alias("last_updated")
        )
    
    # In a real scenario, you would calculate sentiment here
    # For now, we'll use a placeholder
    campaign_performance = campaign_performance.withColumn(
        "avg_sentiment", 
        F.lit(0.75)  # Placeholder for actual sentiment analysis
    )
    
    # Select final columns
    campaign_performance = campaign_performance.select(
        F.col("campaign").alias("campaign_id"),
        "total_messages",
        "unique_users",
        "avg_sentiment",
        "last_updated"
    )
    
    return campaign_performance

def calculate_daily_activity(df: DataFrame) -> DataFrame:
    """Calculate daily user activity metrics."""
    logger.info("Calculating daily activity metrics...")
    
    # Extract date from timestamp
    daily_activity = df.withColumn("activity_date", F.to_date("timestamp"))
    
    # Group by date and user
    daily_activity = daily_activity.groupBy("activity_date", "userid") \
        .agg(
            F.count("*").alias("message_count"),
            F.collect_set("campaign").alias("campaigns_interacted")
        )
    
    # Select final columns
    daily_activity = daily_activity.select(
        "activity_date",
        F.col("userid").alias("user_id"),
        "message_count",
        "campaigns_interacted"
    )
    
    return daily_activity

def save_analytics(df: DataFrame, table_name: str, db_config: Dict, mode: str = "overwrite"):
    """Save analytics data to PostgreSQL."""
    logger.info(f"Saving {table_name} to PostgreSQL...")

    try:
        engine = create_engine(
            f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )

        pdf = df.toPandas()
        for col in pdf.columns:
            if pdf[col].apply(lambda v: isinstance(v, (list, dict))).any():
                pdf[col] = pdf[col].apply(lambda v: json.dumps(v) if v is not None else None)

        if_exists = "replace" if mode == "overwrite" else "append"
        pdf.to_sql(table_name, engine, if_exists=if_exists, index=False)

        logger.info(f"Successfully saved {len(pdf)} records to {table_name}")
    except Exception as e:
        logger.error(f"Error saving to {table_name}: {e}")
        raise

def generate_analytics_report(spark: SparkSession, output_path: str):
    """Generate a summary analytics report in JSON format."""
    logger.info("Generating analytics report...")

    raise RuntimeError("generate_analytics_report() now requires pipeline DataFrames. Call generate_analytics_report_from_dfs().")


def generate_analytics_report_from_dfs(
    df_raw: DataFrame,
    user_engagement: DataFrame,
    campaign_performance: DataFrame,
    daily_activity: DataFrame,
    output_path: str,
):
    """Generate a summary analytics report in JSON format from Spark DataFrames."""
    logger.info("Generating analytics report...")

    total_messages = df_raw.count()
    total_users = df_raw.select("userid").distinct().count()
    total_campaigns = df_raw.select("campaign").distinct().count()

    engagement_numeric = user_engagement.select(
        F.when(F.col("engagement_score") == "high", F.lit(3))
        .when(F.col("engagement_score") == "medium", F.lit(2))
        .otherwise(F.lit(1))
        .alias("engagement_numeric")
    )
    avg_engagement_score = engagement_numeric.select(F.avg("engagement_numeric").alias("avg")).first()["avg"]
    avg_engagement_score = float(avg_engagement_score) if avg_engagement_score is not None else 0.0

    most_engaged_users = (
        user_engagement.orderBy(F.desc("message_count"))
        .limit(5)
        .select("user_id", "message_count", "engagement_score", "preferred_campaigns", "last_active")
        .toPandas()
        .to_dict("records")
    )

    best_performing_campaigns = (
        campaign_performance.orderBy(F.desc("total_messages"))
        .limit(5)
        .toPandas()
        .to_dict("records")
    )

    last_7 = df_raw.withColumn("activity_date", F.to_date("timestamp"))
    messages_last_7_days_rows = (
        last_7.groupBy("activity_date")
        .agg(F.count("*").alias("message_count"))
        .orderBy(F.desc("activity_date"))
        .limit(7)
        .toPandas()
        .to_dict("records")
    )
    messages_last_7_days = {
        str(r["activity_date"]): int(r["message_count"]) for r in messages_last_7_days_rows
    }

    user_activity_rows = (
        daily_activity.groupBy("activity_date")
        .agg(
            F.countDistinct("user_id").alias("active_users"),
            F.sum("message_count").alias("messages"),
        )
        .orderBy(F.desc("activity_date"))
        .limit(7)
        .toPandas()
        .to_dict("records")
    )
    user_activity = {
        str(r["activity_date"]): {
            "active_users": int(r["active_users"]),
            "messages": int(r["messages"]) if r["messages"] is not None else 0,
        }
        for r in user_activity_rows
    }

    report = {
        "report_timestamp": datetime.utcnow().isoformat(),
        "summary": {
            "total_users": int(total_users),
            "total_campaigns": int(total_campaigns),
            "total_messages": int(total_messages),
            "avg_engagement_score": avg_engagement_score,
        },
        "top_performers": {
            "most_engaged_users": most_engaged_users,
            "best_performing_campaigns": best_performing_campaigns,
        },
        "trends": {
            "messages_last_7_days": messages_last_7_days,
            "user_activity": user_activity,
        },
    }
    
    def _json_default(o):
        if isinstance(o, (datetime,)):
            return o.isoformat()
        if isinstance(o, (pd.Timestamp,)):
            return o.isoformat()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return str(o)

    # Save report to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=_json_default)
    
    logger.info(f"Analytics report saved to {output_path}")
    return report

def aggregate_metrics(spark: SparkSession, df: DataFrame, db_config: Dict = None):
    """
    Main function to calculate and store all analytics metrics.
    
    Args:
        spark: Active SparkSession
        df: Input DataFrame with user interactions
        db_config: Database configuration dictionary (can include 'type' for sqlite/postgresql)
    """
    if db_config is None:
        # Default to SQLite for simplicity
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
    
    try:
        # Use the new AnalyticsDB module for consistency
        from src.pipeline.analytics_db import AnalyticsDB
        analytics_db = AnalyticsDB(db_config)
        analytics_db.create_tables()
        
        # Calculate metrics
        user_engagement = calculate_user_engagement(df)
        campaign_performance = calculate_campaign_performance(df)
        daily_activity = calculate_daily_activity(df)
        
        # Save using AnalyticsDB
        analytics_db.save_dataframe(user_engagement, "user_engagement", "overwrite")
        analytics_db.save_dataframe(campaign_performance, "campaign_performance", "overwrite")
        analytics_db.save_dataframe(daily_activity, "user_activity_daily", "append")
        
        # Generate and save report
        report_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data",
            "reports",
            "analytics_report.json",
        )
        generate_analytics_report_from_dfs(
            df_raw=df,
            user_engagement=user_engagement,
            campaign_performance=campaign_performance,
            daily_activity=daily_activity,
            output_path=report_path,
        )
        
        # Export CSVs
        reports_dir = os.path.dirname(report_path)
        user_engagement.toPandas().to_csv(os.path.join(reports_dir, "user_engagement.csv"), index=False)
        campaign_performance.toPandas().to_csv(os.path.join(reports_dir, "campaign_performance.csv"), index=False)
        daily_activity.toPandas().to_csv(os.path.join(reports_dir, "daily_activity.csv"), index=False)
        logger.info(f"CSV reports saved to {reports_dir}")
        
        logger.info("Analytics pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Error in analytics pipeline: {e}")
        raise

if __name__ == "__main__":
    # For local testing
    from pyspark.sql import SparkSession
    from ingestion import ingest_data
    
    # Initialize Spark
    spark = SparkSession.builder \
        .appName("AnalyticsPipeline") \
        .config("spark.jars", "/path/to/postgresql-42.6.0.jar") \
        .getOrCreate()  # Update with actual path
    
    try:
        # Test the analytics pipeline
        df = ingest_data(spark)
        aggregate_metrics(spark, df)
    finally:
        spark.stop()
