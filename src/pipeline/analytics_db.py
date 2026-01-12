"""
Analytics database module supporting both PostgreSQL and SQLite.
"""
import os
import logging
from typing import Dict, Optional
from sqlalchemy import create_engine, text
# SQLAlchemy 1.4 compatibility
try:
    from sqlalchemy.engine import Engine
    SQLALCHEMY_2 = False
except ImportError:
    SQLALCHEMY_2 = True
from pyspark.sql import DataFrame
import pandas as pd
import json

logger = logging.getLogger(__name__)


class AnalyticsDB:
    """Analytics database handler supporting PostgreSQL and SQLite."""
    
    def __init__(self, db_config: Dict = None):
        if db_config is None:
            db_config = {}
        
        # Determine database type
        db_type = db_config.get('type', os.getenv('ANALYTICS_DB_TYPE', 'postgresql')).lower()
        
        if db_type == 'sqlite':
            self.db_type = 'sqlite'
            db_path = db_config.get('path', os.getenv('ANALYTICS_DB_PATH', 'analytics.db'))
            self.engine = create_engine(f'sqlite:///{db_path}')
            logger.info(f"Using SQLite database: {db_path}")
        else:
            # PostgreSQL
            self.db_type = 'postgresql'
            host = db_config.get('host', os.getenv('POSTGRES_HOST', 'localhost'))
            port = db_config.get('port', os.getenv('POSTGRES_PORT', '5432'))
            database = db_config.get('database', os.getenv('POSTGRES_DB', 'analytics'))
            user = db_config.get('user', os.getenv('POSTGRES_USER', 'user'))
            password = db_config.get('password', os.getenv('POSTGRES_PASSWORD', 'password'))
            
            self.engine = create_engine(
                f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}'
            )
            logger.info(f"Using PostgreSQL database: {host}:{port}/{database}")
    
    def create_tables(self):
        """Create necessary tables if they don't exist."""
        try:
            with self.engine.begin() as conn:
                if self.db_type == 'sqlite':
                    # SQLite syntax
                    conn.execute(text("""
                        CREATE TABLE IF NOT EXISTS user_engagement (
                            user_id TEXT,
                            message_count INTEGER,
                            engagement_score TEXT,
                            last_active TIMESTAMP,
                            preferred_campaigns TEXT,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """))
                    conn.execute(text("""
                        CREATE TABLE IF NOT EXISTS campaign_performance (
                            campaign_id TEXT,
                            total_messages INTEGER,
                            unique_users INTEGER,
                            avg_sentiment REAL,
                            last_updated TIMESTAMP,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """))
                    conn.execute(text("""
                        CREATE TABLE IF NOT EXISTS user_activity_daily (
                            activity_date DATE,
                            user_id TEXT,
                            message_count INTEGER,
                            campaigns_interacted TEXT,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """))
                    conn.execute(text("""
                        CREATE TABLE IF NOT EXISTS campaign_engagement_frequency (
                            campaign_id TEXT PRIMARY KEY,
                            engagement_count INTEGER,
                            unique_users INTEGER,
                            last_engagement TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """))
                else:
                    # PostgreSQL syntax
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
                    conn.execute(text("""
                        CREATE TABLE IF NOT EXISTS campaign_engagement_frequency (
                            campaign_id TEXT PRIMARY KEY,
                            engagement_count INTEGER,
                            unique_users INTEGER,
                            last_engagement TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """))
            
            logger.info("Analytics tables created/validated successfully")
            
        except Exception as e:
            # Handle PostgreSQL type conflicts - tables might already exist
            if "pg_type_typname_nsp_index" in str(e) or "duplicate" in str(e).lower():
                logger.warning(f"Database type conflict (tables may already exist): {e}")
                # Verify tables exist by attempting to query them
                try:
                    with self.engine.connect() as conn:
                        if self.db_type == 'sqlite':
                            result = conn.execute(text("""
                                SELECT name FROM sqlite_master 
                                WHERE type='table' 
                                AND name IN ('user_engagement', 'campaign_performance', 'user_activity_daily', 'campaign_engagement_frequency')
                            """))
                        else:
                            result = conn.execute(text("""
                                SELECT table_name 
                                FROM information_schema.tables 
                                WHERE table_schema = 'public' 
                                AND table_name IN ('user_engagement', 'campaign_performance', 'user_activity_daily', 'campaign_engagement_frequency')
                            """))
                        existing_tables = [row[0] for row in result]
                        if len(existing_tables) >= 3:  # At least 3 of 4 tables exist
                            logger.info(f"Tables exist: {existing_tables}. Continuing with pipeline.")
                            return
                        else:
                            logger.warning(f"Only {len(existing_tables)} tables found, but continuing anyway")
                except Exception as verify_error:
                    logger.warning(f"Could not verify existing tables: {verify_error}")
                # Continue anyway - tables might work
                logger.warning("Continuing despite table creation error")
            else:
                logger.error(f"Error creating analytics tables: {e}")
                # Don't raise - try to continue as tables might already exist
                logger.warning("Continuing despite error - tables may already exist")
    
    def save_dataframe(self, df: DataFrame, table_name: str, mode: str = "overwrite"):
        """Save DataFrame to database table."""
        logger.info(f"Saving {table_name} to {self.db_type}...")
        
        try:
            pdf = df.toPandas()
            
            # Convert complex types to JSON strings
            for col in pdf.columns:
                if pdf[col].apply(lambda v: isinstance(v, (list, dict))).any():
                    pdf[col] = pdf[col].apply(lambda v: json.dumps(v) if v is not None else None)
            
            if_exists = "replace" if mode == "overwrite" else "append"
            pdf.to_sql(table_name, self.engine, if_exists=if_exists, index=False)
            
            logger.info(f"Successfully saved {len(pdf)} records to {table_name}")
            return len(pdf)
            
        except Exception as e:
            logger.error(f"Error saving to {table_name}: {e}")
            raise
    
    def get_campaign_engagement_frequency(self, campaign_ids: list = None) -> pd.DataFrame:
        """Get engagement frequency for campaigns."""
        try:
            if campaign_ids:
                query = text("""
                    SELECT campaign_id, engagement_count, unique_users, last_engagement
                    FROM campaign_engagement_frequency
                    WHERE campaign_id IN :campaign_ids
                    ORDER BY engagement_count DESC
                """)
                result = pd.read_sql(query, self.engine, params={"campaign_ids": tuple(campaign_ids)})
            else:
                query = text("""
                    SELECT campaign_id, engagement_count, unique_users, last_engagement
                    FROM campaign_engagement_frequency
                    ORDER BY engagement_count DESC
                """)
                result = pd.read_sql(query, self.engine)
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting campaign engagement frequency: {e}")
            return pd.DataFrame()
    
    def update_campaign_engagement_frequency(self, df: DataFrame):
        """Update or insert campaign engagement frequency metrics."""
        try:
            from pyspark.sql import functions as F
            
            # Calculate engagement frequency per campaign
            engagement_df = df.groupBy("campaign") \
                .agg(
                    F.count("*").alias("engagement_count"),
                    F.countDistinct("userid").alias("unique_users"),
                    F.max("timestamp").alias("last_engagement")
                ) \
                .select(
                    F.col("campaign").alias("campaign_id"),
                    "engagement_count",
                    "unique_users",
                    "last_engagement"
                )
            
            # Save to database (upsert logic depends on database type)
            pdf = engagement_df.toPandas()
            
            # For SQLite, use replace mode (upsert via REPLACE INTO or INSERT OR REPLACE)
            # For PostgreSQL, we'd need ON CONFLICT, so we'll use a simple approach
            if self.db_type == 'sqlite':
                # SQLite: use to_sql with if_exists='replace' after clearing campaign_engagement_frequency
                with self.engine.begin() as conn:
                    conn.execute(text("DELETE FROM campaign_engagement_frequency"))
                pdf.to_sql('campaign_engagement_frequency', self.engine, if_exists='append', index=False)
            else:
                # PostgreSQL: upsert using ON CONFLICT
                for _, row in pdf.iterrows():
                    with self.engine.begin() as conn:
                        conn.execute(text("""
                            INSERT INTO campaign_engagement_frequency 
                            (campaign_id, engagement_count, unique_users, last_engagement)
                            VALUES (:campaign_id, :engagement_count, :unique_users, :last_engagement)
                            ON CONFLICT (campaign_id) 
                            DO UPDATE SET 
                                engagement_count = EXCLUDED.engagement_count,
                                unique_users = EXCLUDED.unique_users,
                                last_engagement = EXCLUDED.last_engagement,
                                updated_at = CURRENT_TIMESTAMP
                        """), {
                            "campaign_id": str(row['campaign_id']),
                            "engagement_count": int(row['engagement_count']),
                            "unique_users": int(row['unique_users']),
                            "last_engagement": row['last_engagement']
                        })
            
            logger.info(f"Updated campaign engagement frequency for {len(pdf)} campaigns")
            
        except Exception as e:
            logger.error(f"Error updating campaign engagement frequency: {e}")
            raise


def get_analytics_db(db_config: Dict = None) -> AnalyticsDB:
    """Get an AnalyticsDB instance."""
    return AnalyticsDB(db_config)
