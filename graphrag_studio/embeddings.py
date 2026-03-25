from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


import logging

logger = logging.getLogger(__name__)


from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from .config import Settings

# Standard embedding dimensions for common models
EMBEDDING_DIMENSIONS = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,

}


# Standard embedding dimensions for common models
EMBEDDING_DIMENSIONS = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
    "BAAI/bge-small-en-v1.5": 384,
}



@lru_cache(maxsize=2)
def get_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )



def build_embeddings(settings: Settings) -> HuggingFaceEmbeddings:    """Build HuggingFace embeddings from settings configuration."""
    logger.debug("Building embeddings with model: %s", settings.embedding_model_name)
    """Build HuggingFace embeddings from settings configuration."""
    logger.debug("Building embeddings with model: %s", settings.embedding_model_name)

    return get_embeddings(settings.embedding_model_name)
