# Marketing Personalization Platform

A multi-database data platform for AI-driven marketing personalization.

## Setup

### Option A: Run everything with Docker (recommended)

```bash
cd marketing
docker compose up -d --build
```

### Option B: Run pipeline/API locally (dev)

```bash
cd marketing
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you run locally, you still typically run the backing services via Docker:

```bash
docker compose up -d
```

## Quick Start

### 1) Start services

```bash
cd marketing
docker compose up -d --build
```

### 2) Run the pipeline

#### Run via Airflow (recommended)

1. Open Airflow UI: `http://localhost:8080`
2. Login:
   - Username: `admin`
   - Password: `admin`

3. Trigger the DAG: `marketing_personalization_pipeline`

#### Run locally (without Airflow)

```bash
# First install dependencies (see Option B above)
python -m src.pipeline.main --input data/sample-conversations.json
```

### 3) Test the API

```bash
curl "http://localhost:8000/recommendations/u1?top_k=5"
```

## Service URLs

- **API**: `http://localhost:8000`
- **Airflow**: `http://localhost:8080`
- **Dashboard (Streamlit)**: `http://localhost:8501`
- **Neo4j Browser**: `http://localhost:7474` (user/pass: `neo4j` / `password`)
- **Milvus**: `localhost:19530`
- **Postgres**: `localhost:5432` (db/user/pass: `analytics` / `user` / `password`)
- **MongoDB**: `localhost:27017` (db: `marketing_db`)

## Outputs

- **Analytics report**: Generated after pipeline run (check Airflow logs for location)
- **API response**: Recommendations available at `/recommendations/{user_id}` endpoint
- **(Optional) lineage / monitoring**: `data/reports/` (if enabled by the pipeline)

## Troubleshooting

### Common Issues

1. **Docker permission error with Airflow entrypoint**
   ```bash
   chmod +x src/dags/airflow-entrypoint.sh
   docker compose restart airflow
   ```

2. **Local pipeline fails with ModuleNotFoundError**
   - Ensure you've installed dependencies: `pip install -r requirements.txt`
   - Python version must be 3.10+ (some packages require 3.11+ for Airflow)

3. **Services not accessible**
   - Check all containers are running: `docker ps`
   - Verify ports aren't already in use
   - Restart services: `docker compose restart`

4. **Airflow DAG not visible**
   - Check DAG file location: `src/dags/marketing_personalization_pipeline_dag.py`
   - Verify Airflow can access the mounted volume
   - Check Airflow logs: `docker compose logs airflow`

### Verification Steps

1. **Check all services**: `docker ps`
2. **Test API**: `curl "http://localhost:8000/recommendations/u1?top_k=5"`
3. **Access Airflow**: Open `http://localhost:8080` (admin/admin)
4. **Check DAG status**: In Airflow UI, ensure `marketing_personalization_pipeline` is unpaused

## Design choices (what & why)

- **Spark for batch ETL**: scalable ingestion + embedding generation + analytics aggregations.
- **Milvus for vector search**: fast approximate nearest-neighbor search over message/user embeddings.
- **Neo4j for relationships**: explicit user↔message↔campaign graph for explainability and graph queries.
- **Postgres for analytics**: structured aggregates for API ranking/explanations and reporting.
- **MongoDB for document storage**: raw/enriched conversation documents (flexible schema, easy replay).
- **Redis for caching**: cache-aside to reduce latency and improve tail performance.
- **Airflow orchestration**: reproducible scheduled runs, retries, and operational visibility.

## 🏗️ Architecture
See:

- [Architecture.md](./Architecture.md) (system diagram + trade-offs)
- [architecture-design.md](./architecture-design.md) (detailed architecture notes)

## 📦 Dependencies
- Python 3.10+ (local dev)
- Docker & Docker Compose
- Apache Spark / PySpark 3.5.x
- Milvus 2.4.x
- Neo4j 5.x
- Redis 7.x
- PostgreSQL 15
- MongoDB 7
- Airflow 2.8.x
