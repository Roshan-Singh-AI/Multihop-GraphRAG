from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from .config import Settings


@lru_cache(maxsize=2)
def get_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )



def build_embeddings(settings: Settings) -> HuggingFaceEmbeddings:
    return get_embeddings(settings.embedding_model_name)
