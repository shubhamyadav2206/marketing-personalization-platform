"""
Document storage module for MongoDB.
Stores text + metadata (conversation data) in MongoDB.
"""
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, DuplicateKeyError
from typing import Dict, List, Optional, Any
import logging
import os
import hashlib
from datetime import datetime
from pyspark.sql import DataFrame
import json

logger = logging.getLogger(__name__)

class MongoDBStore:
    """MongoDB storage handler for conversation data."""
    
    def __init__(
        self,
        host: str = None,
        port: int = None,
        database: str = None,
        collection: str = None,
        username: str = None,
        password: str = None,
    ):
        self.host = host or os.getenv("MONGODB_HOST", "localhost")
        self.port = port or int(os.getenv("MONGODB_PORT", "27017"))
        self.database_name = database or os.getenv("MONGODB_DATABASE", "marketing_db")
        self.collection_name = collection or os.getenv("MONGODB_COLLECTION", "conversations")
        self.username = username or os.getenv("MONGODB_USER", None)
        self.password = password or os.getenv("MONGODB_PASSWORD", None)
        
        self.client = None
        self.db = None
        self.collection = None
        
    def connect(self):
        """Establish connection to MongoDB."""
        try:
            if self.username and self.password:
                connection_string = f"mongodb://{self.username}:{self.password}@{self.host}:{self.port}/{self.database_name}?authSource=admin"
            else:
                connection_string = f"mongodb://{self.host}:{self.port}/"
            
            self.client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
            # Test connection
            self.client.admin.command('ping')
            self.db = self.client[self.database_name]
            self.collection = self.db[self.collection_name]
            
            # Create indexes for better query performance
            self.collection.create_index([("userid", 1)])
            self.collection.create_index([("campaign", 1)])
            self.collection.create_index([("timestamp", -1)])
            self.collection.create_index([("processing_timestamp", -1)])
            
            # Create unique index on message_id for idempotency
            try:
                self.collection.create_index([("message_id", 1)], unique=True)
            except Exception as e:
                logger.warning(f"Could not create unique index on message_id: {e}")
            
            logger.info(f"Connected to MongoDB at {self.host}:{self.port}, database: {self.database_name}, collection: {self.collection_name}")
            return True
            
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
        except Exception as e:
            logger.error(f"Error connecting to MongoDB: {e}")
            raise
    
    def close(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
    
    def store_documents(self, df: DataFrame, batch_size: int = 1000):
        """
        Store documents from Spark DataFrame to MongoDB.
        
        Args:
            df: Spark DataFrame containing conversation data
            batch_size: Number of records to process in each batch
        """
        if not self.client:
            self.connect()
        
        logger.info(f"Storing documents to MongoDB collection: {self.collection_name}")
        
        # Convert to pandas for batch processing
        pdf = df.toPandas()
        
        # Prepare documents
        documents = []
        for _, row in pdf.iterrows():
            # Generate unique message ID for idempotency
            message_id_raw = f"{row['userid']}|{row.get('campaign', '')}|{str(row['timestamp'])}|{row['message']}"
            message_id = hashlib.sha256(message_id_raw.encode('utf-8')).hexdigest()
            
            doc = {
                "message_id": message_id,
                "userid": str(row['userid']),
                "message": str(row['message']),
                "timestamp": row['timestamp'] if isinstance(row['timestamp'], datetime) else datetime.fromisoformat(str(row['timestamp']).replace('Z', '+00:00')),
                "campaign": str(row.get('campaign', '')),
                "processing_timestamp": datetime.utcnow(),
                "metadata": {
                    "has_embedding": 'embedding' in row and row.get('embedding') is not None,
                    "embedding_dim": len(row.get('embedding', [])) if 'embedding' in row and row.get('embedding') is not None else 0,
                }
            }
            documents.append(doc)
        
        # Insert documents in batches with upsert to handle duplicates
        total_inserted = 0
        total_updated = 0
        total_errors = 0
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            for doc in batch:
                try:
                    result = self.collection.update_one(
                        {"message_id": doc["message_id"]},
                        {"$set": doc},
                        upsert=True
                    )
                    if result.upserted_id:
                        total_inserted += 1
                    else:
                        total_updated += 1
                except Exception as e:
                    logger.warning(f"Error inserting document {doc.get('message_id', 'unknown')}: {e}")
                    total_errors += 1
            
            if (i // batch_size + 1) % 10 == 0:
                logger.info(f"Processed batch {i // batch_size + 1}: inserted={total_inserted}, updated={total_updated}, errors={total_errors}")
        
        logger.info(f"MongoDB storage completed: inserted={total_inserted}, updated={total_updated}, errors={total_errors}, total={len(documents)}")
        return {"inserted": total_inserted, "updated": total_updated, "errors": total_errors, "total": len(documents)}
    
    def get_documents_by_user(self, user_id: str, limit: int = 100) -> List[Dict]:
        """Retrieve documents for a specific user."""
        if not self.client:
            self.connect()
        
        cursor = self.collection.find({"userid": user_id}).sort("timestamp", -1).limit(limit)
        return list(cursor)
    
    def get_documents_by_campaign(self, campaign_id: str, limit: int = 100) -> List[Dict]:
        """Retrieve documents for a specific campaign."""
        if not self.client:
            self.connect()
        
        cursor = self.collection.find({"campaign": campaign_id}).sort("timestamp", -1).limit(limit)
        return list(cursor)
    
    def get_statistics(self) -> Dict:
        """Get collection statistics."""
        if not self.client:
            self.connect()
        
        total_docs = self.collection.count_documents({})
        unique_users = len(self.collection.distinct("userid"))
        unique_campaigns = len(self.collection.distinct("campaign"))
        
        return {
            "total_documents": total_docs,
            "unique_users": unique_users,
            "unique_campaigns": unique_campaigns,
            "collection_name": self.collection_name,
            "database_name": self.database_name
        }


def store_to_mongodb(df: DataFrame, batch_size: int = 1000):
    """
    Main function to store DataFrame to MongoDB.
    
    Args:
        df: Spark DataFrame containing conversation data
        batch_size: Batch size for insertion
    """
    store = MongoDBStore()
    try:
        store.connect()
        result = store.store_documents(df, batch_size)
        return result
    finally:
        store.close()
