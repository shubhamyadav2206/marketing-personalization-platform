from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, FloatType, IntegerType
import numpy as np
import os
from pymilvus import (
    connections,
    utility,
    FieldSchema, 
    DataType, 
    CollectionSchema, 
    Collection,
)
import json
from typing import Dict, List, Optional, Tuple
import time

# Milvus connection parameters
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = "marketing_embeddings"
EMBEDDING_DIM = 384  # Dimension for all-MiniLM-L6-v2 model

# Define the schema for our vector database
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="campaign_id", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="message", dtype=DataType.VARCHAR, max_length=1000),
    FieldSchema(name="timestamp", dtype=DataType.INT64),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
]

schema = CollectionSchema(fields, "Marketing campaign messages with embeddings")

def connect_to_milvus(host: str = MILVUS_HOST, port: str = MILVUS_PORT):
    """Establish connection to Milvus server."""
    try:
        connections.connect("default", host=host, port=port)
        print(f"Connected to Milvus server at {host}:{port}")
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        raise

def create_collection(collection_name: str = COLLECTION_NAME, overwrite: bool = False):
    """Create a new collection in Milvus."""
    connect_to_milvus()
    
    # Drop collection if it exists and overwrite is True
    if utility.has_collection(collection_name) and overwrite:
        utility.drop_collection(collection_name)
        print(f"Dropped existing collection: {collection_name}")
    
    # Create collection if it doesn't exist
    if not utility.has_collection(collection_name):
        collection = Collection(name=collection_name, schema=schema)
        
        # Create index for faster similarity search
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        
        collection.create_index(field_name="embedding", index_params=index_params)
        print(f"Created collection {collection_name} with index")
        return collection
    else:
        print(f"Using existing collection: {collection_name}")
        return Collection(collection_name)

def store_vectors(df: DataFrame, collection_name: str = COLLECTION_NAME, batch_size: int = 1000):
    """
    Store vector embeddings in Milvus.
    
    Args:
        df: Spark DataFrame containing user messages and embeddings
        collection_name: Name of the Milvus collection
        batch_size: Number of records to insert in each batch
    """
    # Ensure we have the required columns
    required_columns = {"userid", "campaign", "message", "timestamp", "embedding"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Create or get the collection
    collection = create_collection(collection_name, overwrite=True)
    
    # Convert to pandas for batch processing
    pdf = df.select("userid", "campaign", "message", "timestamp", "embedding").toPandas()
    
    # Prepare data for insertion
    entities = []
    for _, row in pdf.iterrows():
        entities.append({
            "user_id": str(row['userid']),
            "campaign_id": str(row['campaign']),
            "message": str(row['message']),
            "timestamp": int(row['timestamp'].timestamp() * 1000),  # Convert to milliseconds
            "embedding": row['embedding']
        })
    
    # Insert data in batches
    total_records = len(entities)
    print(f"Inserting {total_records} records into Milvus...")
    
    for i in range(0, total_records, batch_size):
        batch = entities[i:i + batch_size]
        try:
            insert_result = collection.insert(batch)
            print(f"Inserted batch {i//batch_size + 1}: {len(batch)} records")
        except Exception as e:
            print(f"Error inserting batch {i//batch_size + 1}: {e}")
            raise
    
    # Flush to make sure all changes are synced
    collection.flush()
    
    # Load collection for searching
    collection.load()
    
    # Print collection statistics
    print(f"\nCollection statistics:")
    print(f"Number of entities: {collection.num_entities}")
    
    return collection

def search_similar_vectors(embedding: List[float], top_k: int = 5, collection_name: str = COLLECTION_NAME):
    """
    Search for similar vectors in Milvus.
    
    Args:
        embedding: Query embedding vector
        top_k: Number of similar vectors to return
        collection_name: Name of the Milvus collection
        
    Returns:
        List of similar items with scores
    """
    if not utility.has_collection(collection_name):
        raise ValueError(f"Collection {collection_name} does not exist")
    
    collection = Collection(collection_name)
    collection.load()
    
    # Convert single embedding to list of lists for search
    search_vectors = [embedding]
    
    # Define search parameters
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10}
    }
    
    # Execute search
    results = collection.search(
        data=search_vectors,
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["user_id", "campaign_id", "message"]
    )
    
    # Process and return results
    similar_items = []
    for hits in results:
        for hit in hits:
            item = {
                "id": hit.id,
                "score": hit.score,
                "user_id": hit.entity.get('user_id'),
                "campaign_id": hit.entity.get('campaign_id'),
                "message": hit.entity.get('message')
            }
            similar_items.append(item)
    
    return similar_items

def delete_collection(collection_name: str = COLLECTION_NAME):
    """Delete a collection from Milvus."""
    connect_to_milvus()
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        print(f"Dropped collection: {collection_name}")
    else:
        print(f"Collection {collection_name} does not exist")
