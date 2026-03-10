"""GraphRAG Studio package."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .service import GraphRAGService

__all__ = ["GraphRAGService"]


def __getattr__(name: str):
    if name == "GraphRAGService":
        from .service import GraphRAGService

        return GraphRAGService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
