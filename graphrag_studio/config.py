from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


from functools import lru_cache

from dotenv import load_dotenv
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "GraphRAG Studio"
    env: str = "dev"

    groq_api_key: SecretStr | None = None
    groq_answer_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    groq_extraction_model: str = "openai/gpt-oss-20b"
    groq_max_retries: int = 2

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: SecretStr = Field(default=SecretStr("password12345"))
    neo4j_database: str = "neo4j"

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_index_name: str = "chunk_embeddings"
    keyword_index_name: str = "chunk_keywords"
    vector_similarity_function: str = "cosine"

    chunk_size: int = 900
    chunk_overlap: int = 120
    retrieval_top_k: int = 6
    graph_max_hops: int = 2
    graph_neighbor_limit: int = 18
    query_seed_entity_limit: int = 8
    chunk_entity_limit: int = 6


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
