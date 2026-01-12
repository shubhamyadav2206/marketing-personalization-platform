from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator


with DAG(
    dag_id="marketing_personalization_pipeline",
    description="Run the marketing personalization Spark pipeline (ingestion -> embeddings -> Milvus/Neo4j -> analytics report)",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    default_args={
        "owner": "airflow",
        "retries": 1,
        "retry_delay": timedelta(minutes=2),
    },
    tags=["pipeline", "spark", "marketing"],
) as dag:
    run_pipeline = BashOperator(
        task_id="run_pipeline",
        bash_command=(
            "set -euo pipefail; "
            "cd /opt/airflow/project; "
            "export JAVA_HOME=$(find /usr/lib/jvm -maxdepth 1 -type d -name 'java-17-openjdk-*' 2>/dev/null | head -1 || echo '/usr/lib/jvm/java-17-openjdk-arm64'); "
            "export PATH=$JAVA_HOME/bin:$PATH; "
            "export MILVUS_HOST=milvus; "
            "export MILVUS_PORT=19530; "
            "export NEO4J_URI=bolt://neo4j:7687; "
            "export NEO4J_USER=neo4j; "
            "export NEO4J_PASSWORD=password; "
            "export POSTGRES_HOST=postgres; "
            "export POSTGRES_PORT=5432; "
            "export POSTGRES_DB=analytics; "
            "export POSTGRES_USER=user; "
            "export POSTGRES_PASSWORD=password; "
            "export MONGODB_HOST=mongodb; "
            "export MONGODB_PORT=27017; "
            "export MONGODB_DATABASE=marketing_db; "
            "export ANALYTICS_DB_TYPE=postgresql; "
            "export PYTHONUNBUFFERED=1; "
            "python -m src.pipeline.main --input /opt/airflow/project/data/sample-conversations.json"
        ),
        env={
            "MILVUS_HOST": "milvus",
            "MILVUS_PORT": "19530",
            "NEO4J_URI": "bolt://neo4j:7687",
            "NEO4J_USER": "neo4j",
            "NEO4J_PASSWORD": "password",
            "POSTGRES_HOST": "postgres",
            "POSTGRES_PORT": "5432",
            "POSTGRES_DB": "analytics",
            "POSTGRES_USER": "user",
            "POSTGRES_PASSWORD": "password",
            "MONGODB_HOST": "mongodb",
            "MONGODB_PORT": "27017",
            "MONGODB_DATABASE": "marketing_db",
            "ANALYTICS_DB_TYPE": "postgresql",
            "PYTHONUNBUFFERED": "1",
        },
    )

    run_pipeline
