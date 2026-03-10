# Deployment Guide

## Local with Docker Compose

```bash
docker compose up --build
```

That starts:

- Neo4j on `localhost:7474` and `localhost:7687`
- Streamlit UI on `localhost:8501`
- FastAPI on `localhost:8000`

## Managed setup

A good portfolio deployment pattern is:

- **Neo4j AuraDB** for the graph database
- **one small Python app** for Streamlit
- optional **FastAPI deployment** if you want a public API surface

## Environment variables

Required:

- `GROQ_API_KEY`
- `NEO4J_URI`
- `NEO4J_USERNAME`
- `NEO4J_PASSWORD`

Common overrides:

- `GROQ_ANSWER_MODEL`
- `GROQ_EXTRACTION_MODEL`
- `EMBEDDING_MODEL_NAME`
- `RETRIEVAL_TOP_K`
- `GRAPH_MAX_HOPS`

## Production notes

For a real deployment, pin your container images and lock your dependencies before shipping.
