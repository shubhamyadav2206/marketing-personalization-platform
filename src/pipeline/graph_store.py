from neo4j import GraphDatabase
from typing import Dict, List, Optional, Any
import logging
import os
from pyspark.sql import DataFrame
import time
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Neo4jConnection:
    """A context manager for Neo4j database connections."""
    
    def __init__(self, uri: str, user: str, password: str):
        self._uri = uri
        self._user = user
        self._password = password
        self._driver = None
    
    def __enter__(self):
        self._driver = GraphDatabase.driver(self._uri, auth=(self._user, self._password))
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._driver is not None:
            self._driver.close()
    
    def execute_query(self, query: str, parameters: Dict = None, db: str = None) -> List[Dict[str, Any]]:
        """Execute a read query and return results."""
        with self._driver.session(database=db) as session:
            result = session.run(query, parameters or {})
            return [dict(record) for record in result]
    
    def execute_write(self, query: str, parameters: Dict = None, db: str = None):
        """Execute a write query."""
        with self._driver.session(database=db).begin_transaction() as tx:
            result = tx.run(query, parameters or {})
            tx.commit()
            return result

def init_neo4j_connection(
    uri: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
) -> Neo4jConnection:
    """Initialize and return a Neo4j connection."""
    uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = user or os.getenv("NEO4J_USER", "neo4j")
    password = password or os.getenv("NEO4J_PASSWORD", "password")
    try:
        conn = Neo4jConnection(uri, user, password)
        # Test the connection
        with conn:
            conn.execute_query("RETURN 1")
            logger.info("Successfully connected to Neo4j")
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {e}")
        raise

def create_constraints(conn: Neo4jConnection):
    """Create necessary constraints in Neo4j."""
    constraints = [
        "CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.userId IS UNIQUE",
        "CREATE CONSTRAINT campaign_id IF NOT EXISTS FOR (c:Campaign) REQUIRE c.campaignId IS UNIQUE",
        "CREATE CONSTRAINT message_id IF NOT EXISTS FOR (m:Message) REQUIRE m.messageId IS UNIQUE"
    ]
    
    for constraint in constraints:
        try:
            conn.execute_write(constraint)
            logger.info(f"Created constraint: {constraint}")
        except Exception as e:
            logger.warning(f"Failed to create constraint {constraint}: {e}")

def build_graph(df: DataFrame, batch_size: int = 1000):
    """
    Build a knowledge graph from the DataFrame.
    
    Args:
        df: Spark DataFrame containing user interactions
        batch_size: Number of records to process in each batch
    """
    logger.info("Starting graph construction...")
    start_time = time.time()
    
    # Initialize Neo4j connection
    conn = init_neo4j_connection()
    
    try:
        # Create constraints
        create_constraints(conn)
        
        # Process data in batches
        total_records = df.count()
        logger.info(f"Processing {total_records} records in batches of {batch_size}")
        
        # Convert to pandas for batch processing
        pdf = df.select("userid", "campaign", "message", "timestamp").toPandas()
        
        # Add a stable unique message ID so reruns are idempotent
        def _make_message_id(row) -> str:
            raw = f"{row['userid']}|{row['campaign']}|{row['timestamp']}|{row['message']}"
            return hashlib.sha256(raw.encode("utf-8")).hexdigest()

        pdf["messageId"] = pdf.apply(_make_message_id, axis=1)
        
        # Process each batch
        for i in range(0, len(pdf), batch_size):
            batch = pdf.iloc[i:i + batch_size]
            
            # Create users
            users = batch[['userid']].drop_duplicates()
            user_query = """
            UNWIND $users AS user
            MERGE (u:User {userId: user.userid})
            """
            conn.execute_write(user_query, {"users": users.to_dict('records')})
            
            # Create campaigns
            campaigns = batch[['campaign']].drop_duplicates()
            campaign_query = """
            UNWIND $campaigns AS campaign
            MERGE (c:Campaign {campaignId: campaign.campaign})
            """
            conn.execute_write(campaign_query, {"campaigns": campaigns.to_dict('records')})
            
            # Create messages and relationships (idempotent)
            for _, row in batch.iterrows():
                message_query = """
                MERGE (u:User {userId: $userId})
                MERGE (c:Campaign {campaignId: $campaignId})
                MERGE (m:Message {messageId: $messageId})
                SET m.text = $text,
                    m.timestamp = $timestamp
                MERGE (u)-[:SENT]->(m)
                MERGE (m)-[:ABOUT]->(c)
                """
                conn.execute_write(
                    message_query,
                    {
                        "userId": str(row['userid']),
                        "campaignId": str(row['campaign']),
                        "messageId": str(row['messageId']),
                        "text": str(row['message']),
                        "timestamp": int(row['timestamp'].timestamp() * 1000)
                    }
                )
            
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(pdf)-1)//batch_size + 1}")
        
        # Create similarity relationships between users based on common campaigns
        similarity_query = """
        MATCH (u1:User)-[:SENT]->(m1:Message)-[:ABOUT]->(c:Campaign),
              (u2:User)-[:SENT]->(m2:Message)-[:ABOUT]->(c)
        WHERE u1 <> u2
        WITH u1, u2, COUNT(DISTINCT c) AS common_campaigns
        MERGE (u1)-[s:SIMILAR_TO]-(u2)
        SET s.weight = common_campaigns
        """
        conn.execute_write(similarity_query)
        
        # Calculate and store user engagement metrics
        engagement_query = """
        MATCH (u:User)-[r:SENT]->(m:Message)
        WITH u, COUNT(m) AS message_count
        SET u.messageCount = message_count,
            u.engagementScore = CASE 
                WHEN message_count > 10 THEN 'high'
                WHEN message_count > 5 THEN 'medium'
                ELSE 'low'
            END
        """
        conn.execute_write(engagement_query)
        
        # Create index for faster lookups
        index_queries = [
            "CREATE INDEX user_engagement IF NOT EXISTS FOR (u:User) ON (u.engagementScore)",
            "CREATE INDEX message_timestamp IF NOT EXISTS FOR (m:Message) ON (m.timestamp)"
        ]
        
        for query in index_queries:
            try:
                conn.execute_write(query)
            except Exception as e:
                logger.warning(f"Failed to create index: {e}")
        
        duration = time.time() - start_time
        logger.info(f"Graph construction completed in {duration:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error building graph: {e}")
        raise
    finally:
        # The connection is managed by the context manager
        pass

def get_similar_users(user_id: str, top_k: int = 5) -> List[Dict]:
    """
    Find users similar to the given user based on campaign interactions.
    
    Args:
        user_id: The user ID to find similar users for
        top_k: Number of similar users to return
        
    Returns:
        List of similar users with their similarity scores
    """
    query = """
    MATCH (u1:User {userId: $userId})-[s:SIMILAR_TO]-(u2:User)
    RETURN u2.userId AS userId, s.weight AS similarity
    ORDER BY s.weight DESC
    LIMIT $top_k
    """
    
    try:
        with init_neo4j_connection() as conn:
            results = conn.execute_query(query, {"userId": user_id, "top_k": top_k})
            return [dict(record) for record in results]
    except Exception as e:
        logger.error(f"Error finding similar users: {e}")
        return []

def get_user_engagement(user_id: str) -> Dict:
    """
    Get engagement metrics for a user.
    
    Args:
        user_id: The user ID to get metrics for
        
    Returns:
        Dictionary containing engagement metrics
    """
    query = """
    MATCH (u:User {userId: $userId})
    OPTIONAL MATCH (u)-[:SENT]->(m:Message)
    RETURN {
        userId: u.userId,
        messageCount: u.messageCount,
        engagementScore: u.engagementScore,
        lastActivity: m.timestamp
    } AS metrics
    ORDER BY m.timestamp DESC
    LIMIT 1
    """
    
    try:
        with init_neo4j_connection() as conn:
            result = conn.execute_query(query, {"userId": user_id})
            return result[0]["metrics"] if result else {}
    except Exception as e:
        logger.error(f"Error getting user engagement: {e}")
        return {}

def clear_graph():
    """Clear all data from the graph database."""
    query = "MATCH (n) DETACH DELETE n"
    try:
        with init_neo4j_connection() as conn:
            conn.execute_write(query)
            logger.info("Cleared all data from the graph database")
    except Exception as e:
        logger.error(f"Error clearing graph: {e}")
        raise
