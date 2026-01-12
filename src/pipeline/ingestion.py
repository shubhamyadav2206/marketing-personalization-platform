from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import json
from typing import Dict, List
from pyspark.sql.types import StructType, StructField, StringType, TimestampType
from datetime import datetime
import os


def load_json_data(spark: SparkSession, file_path: str):
    """Load JSON data from the specified file path."""
    try:
        # Define schema for better type safety
        schema = StructType([
            StructField("userid", StringType(), False),
            StructField("message", StringType(), False),
            StructField("timestamp", StringType(), False),
            StructField("campaign", StringType(), True)
        ])
        
        # Read JSON with schema
        # Note: sample-conversations.json is a JSON array (multiline), so we enable multiLine.
        df = spark.read.schema(schema).option("multiLine", True).json(file_path)
        
        # Convert timestamp string to timestamp type
        df = df.withColumn("timestamp", F.to_timestamp("timestamp", "yyyy-MM-dd'T'HH:mm:ssX"))
        
        # Add processing timestamp
        df = df.withColumn("processing_timestamp", F.current_timestamp())
        
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise


def clean_data(df):
    """Clean and preprocess the input DataFrame."""
    # Remove duplicate rows
    df = df.dropDuplicates()
    
    # Filter out rows with null userid or message
    df = df.filter(
        (F.col("userid").isNotNull()) & 
        (F.col("message").isNotNull())
    )
    
    # Truncate message if too long (optional)
    max_length = 1000
    df = df.withColumn("message", 
                      F.when(
                          F.length("message") > max_length, 
                          F.substring("message", 1, max_length)
                      ).otherwise(F.col("message")))
    
    return df


def ingest_data(spark: SparkSession, file_path: str = None):
    """Main function to ingest and process data.
    
    Args:
        spark: Active SparkSession
        file_path: Path to the JSON data file. If None, uses default path.
    """
    if file_path is None:
        # Default path relative to project root
        file_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data",
            "sample-conversations.json"
        )
    
    print(f"Loading data from: {file_path}")
    
    # Load and clean data
    df = load_json_data(spark, file_path)
    df_cleaned = clean_data(df)
    
    # Show sample data
    print("Sample data after ingestion:")
    df_cleaned.show(5, truncate=False)
    
    return df_cleaned


if __name__ == "__main__":
    # For local testing
    spark = SparkSession.builder \
        .appName("DataIngestion") \
        .getOrCreate()
    
    try:
        df = ingest_data(spark)
        print(f"Successfully ingested {df.count()} records")
    finally:
        spark.stop()
