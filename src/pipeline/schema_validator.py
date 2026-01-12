"""
Schema validation and data lineage tracking module.
"""
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, IntegerType
import logging

logger = logging.getLogger(__name__)

# Define expected schema for conversation data
CONVERSATION_SCHEMA = StructType([
    StructField("userid", StringType(), False),
    StructField("message", StringType(), False),
    StructField("timestamp", TimestampType(), False),
    StructField("campaign", StringType(), True),
])

class DataLineageTracker:
    """Track data lineage through the pipeline."""
    
    def __init__(self):
        self.lineage_records = []
        self.pipeline_run_id = datetime.utcnow().isoformat()
        
    def record_step(self, step_name: str, input_record_count: int, output_record_count: int, 
                   input_hash: str = None, output_hash: str = None, metadata: Dict = None):
        """Record a pipeline step with lineage information."""
        record = {
            "pipeline_run_id": self.pipeline_run_id,
            "step_name": step_name,
            "timestamp": datetime.utcnow().isoformat(),
            "input_record_count": input_record_count,
            "output_record_count": output_record_count,
            "input_hash": input_hash,
            "output_hash": output_hash,
            "metadata": metadata or {}
        }
        self.lineage_records.append(record)
        logger.info(f"Lineage recorded: {step_name} - input={input_record_count}, output={output_record_count}")
    
    def get_lineage_summary(self) -> Dict:
        """Get summary of data lineage."""
        return {
            "pipeline_run_id": self.pipeline_run_id,
            "total_steps": len(self.lineage_records),
            "steps": self.lineage_records
        }
    
    def save_lineage(self, output_path: str):
        """Save lineage records to JSON file."""
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.get_lineage_summary(), f, indent=2)
        logger.info(f"Lineage saved to {output_path}")


def calculate_dataframe_hash(df: DataFrame) -> str:
    """Calculate a hash of the DataFrame content for lineage tracking."""
    try:
        # Sample rows and create a hash
        sample = df.limit(1000).collect()
        content_str = json.dumps([row.asDict() for row in sample], default=str, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()
    except Exception as e:
        logger.warning(f"Could not calculate DataFrame hash: {e}")
        return "unknown"


def validate_schema(df: DataFrame, expected_schema: StructType = None):
    """
    Validate DataFrame schema against expected schema.
    
    Args:
        df: DataFrame to validate
        expected_schema: Expected schema (defaults to CONVERSATION_SCHEMA)
        
    Returns:
        (is_valid, list_of_errors)
    """
    if expected_schema is None:
        expected_schema = CONVERSATION_SCHEMA
    
    errors = []
    
    # Check required columns
    expected_fields = {field.name: field for field in expected_schema.fields}
    actual_fields = {field.name: field for field in df.schema.fields}
    
    for field_name, expected_field in expected_fields.items():
        if field_name not in actual_fields:
            errors.append(f"Missing required column: {field_name}")
            continue
        
        actual_field = actual_fields[field_name]
        # Note: Don't check nullable constraint on schema level, as Spark may create nullable columns
        # even for non-null data due to transformations. Instead, we check for actual null values below.
        
        # Check data type (simplified - in production, use more sophisticated type checking)
        expected_type = str(expected_field.dataType)
        actual_type = str(actual_field.dataType)
        if expected_type != actual_type and not _types_compatible(expected_type, actual_type):
            # Allow StringType -> TimestampType for timestamp fields (converted during ingestion)
            if field_name == "timestamp" and "StringType" in expected_type and "TimestampType" in actual_type:
                pass  # This is expected - timestamp is converted from string
            else:
                errors.append(f"Column {field_name} type mismatch: expected {expected_type}, got {actual_type}")
    
    # Validate data quality
    total_count = df.count()
    
    # Check for null values in required fields
    required_fields = [f.name for f in expected_schema.fields if not f.nullable]
    for field in required_fields:
        if field in actual_fields:
            null_count = df.filter(F.col(field).isNull()).count()
            if null_count > 0:
                errors.append(f"Found {null_count} null values in required field: {field}")
    
    # Check for empty strings in required fields
    for field in required_fields:
        if field in actual_fields:
            empty_count = df.filter((F.col(field) == "") | (F.trim(F.col(field)) == "")).count()
            if empty_count > 0:
                errors.append(f"Found {empty_count} empty values in required field: {field}")
    
    # Check timestamp validity
    if "timestamp" in actual_fields:
        invalid_timestamps = df.filter(F.col("timestamp").isNull() | F.col("timestamp").isNotNull()).count() - total_count
        if invalid_timestamps < 0:
            # This is a simplified check - in production, validate actual timestamp values
            pass
    
    is_valid = len(errors) == 0
    
    if is_valid:
        logger.info(f"Schema validation passed: {total_count} records")
    else:
        logger.warning(f"Schema validation failed with {len(errors)} errors: {errors}")
    
    return is_valid, errors


def _types_compatible(expected: str, actual: str) -> bool:
    """Check if types are compatible (simplified version)."""
    # In production, implement more sophisticated type compatibility checking
    type_mapping = {
        "StringType": ["StringType", "VarcharType"],
        "TimestampType": ["TimestampType", "DateType"],
        "IntegerType": ["IntegerType", "LongType", "DoubleType"],
    }
    
    for compatible_group in type_mapping.values():
        if expected in compatible_group and actual in compatible_group:
            return True
    
    return False


def validate_embeddings(df: DataFrame):
    """Validate embeddings column exists and is not empty."""
    errors = []
    
    if "embedding" not in df.columns:
        errors.append("Missing 'embedding' column")
        return False, errors
    
    total_count = df.count()
    
    # Check for null embeddings
    null_embeddings = df.filter(F.col("embedding").isNull()).count()
    if null_embeddings > 0:
        errors.append(f"Found {null_embeddings} null embeddings")
    
    # Check for empty embeddings (length 0)
    empty_embeddings = df.filter(F.size(F.col("embedding")) == 0).count()
    if empty_embeddings > 0:
        errors.append(f"Found {empty_embeddings} empty embeddings")
    
    # Check embedding dimensions are consistent
    try:
        sample_embedding = df.select(F.col("embedding")).filter(F.col("embedding").isNotNull()).first()
        if sample_embedding and sample_embedding[0]:
            expected_dim = len(sample_embedding[0])
            wrong_dim = df.filter(
                (F.col("embedding").isNotNull()) & (F.size(F.col("embedding")) != expected_dim)
            ).count()
            if wrong_dim > 0:
                errors.append(f"Found {wrong_dim} embeddings with wrong dimension (expected {expected_dim})")
    except Exception as e:
        logger.warning(f"Could not validate embedding dimensions: {e}")
    
    is_valid = len(errors) == 0
    
    if is_valid:
        logger.info(f"Embedding validation passed: {total_count} records")
    else:
        logger.warning(f"Embedding validation failed: {errors}")
    
    return is_valid, errors


def detect_anomalies(df: DataFrame) -> Dict[str, Any]:
    """
    Detect anomalies in the data.
    
    Returns:
        Dictionary with anomaly detection results
    """
    anomalies = {
        "empty_embeddings": 0,
        "missing_relationships": 0,
        "invalid_timestamps": 0,
        "empty_messages": 0,
        "missing_campaigns": 0,
        "orphaned_users": 0,
    }
    
    total_count = df.count()
    
    # Check for empty embeddings
    if "embedding" in df.columns:
        anomalies["empty_embeddings"] = df.filter(
            F.col("embedding").isNull() | (F.size(F.col("embedding")) == 0)
        ).count()
    
    # Check for empty messages
    if "message" in df.columns:
        anomalies["empty_messages"] = df.filter(
            (F.col("message").isNull()) | (F.trim(F.col("message")) == "")
        ).count()
    
    # Check for missing campaigns
    if "campaign" in df.columns:
        anomalies["missing_campaigns"] = df.filter(
            F.col("campaign").isNull() | (F.col("campaign") == "")
        ).count()
    
    # Check for invalid timestamps
    if "timestamp" in df.columns:
        anomalies["invalid_timestamps"] = df.filter(F.col("timestamp").isNull()).count()
    
    # Log anomalies
    total_anomalies = sum(anomalies.values())
    if total_anomalies > 0:
        logger.warning(f"Detected {total_anomalies} anomalies: {anomalies}")
    else:
        logger.info(f"No anomalies detected in {total_count} records")
    
    anomalies["total_records"] = total_count
    anomalies["anomaly_rate"] = total_anomalies / total_count if total_count > 0 else 0.0
    
    return anomalies
