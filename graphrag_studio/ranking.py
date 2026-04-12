from __future__ import annotations

from collections import defaultdict
from typing import Iterable

# Type aliases for score maps
ChunkScoreMap = dict[str, float]
RankMap = dict[str, int]

RRF_K = 60

# Minimum score threshold for including results
MIN_RELEVANCE_SCORE = 0.01

# Score fusion weights (should sum to 1.0)
VECTOR_WEIGHT = 0.55
KEYWORD_WEIGHT = 0.15
GRAPH_WEIGHT = 0.20
SUPPORT_WEIGHT = 0.10


def reciprocal_rank(rank: int | None, k: int = RRF_K) -> float:
    """Calculate reciprocal rank score using RRF formula.
    
    Args:
        rank: Position in ranked list (1-indexed). None or <= 0 returns 0.
        k: Smoothing constant for RRF (default: 60).
        
    Returns:
        Reciprocal rank score between 0 and 1.
    """
    if rank is None or rank <= 0:
        return 0.0
    return 1.0 / (k + rank)


def reciprocal_rank_for_docs(doc_ids: Iterable[str], expected_doc_ids: set[str]) -> float:
    for index, doc_id in enumerate(doc_ids, start=1):
        if doc_id in expected_doc_ids:
            return 1.0 / index
    return 0.0


def hit_at_k(doc_ids: Iterable[str], expected_doc_ids: set[str]) -> bool:
    return any(doc_id in expected_doc_ids for doc_id in doc_ids)


def fuse_candidate_scores(
    vector_ranks: dict[str, int],
    keyword_ranks: dict[str, int],
    graph_ranks: dict[str, int],
    graph_support: dict[str, float],
) -> dict[str, float]:
    """Fuse multiple ranking signals using Reciprocal Rank Fusion.
    
    Combines vector similarity, keyword matching, graph connectivity,
    and support scores into a single relevance score.
    """
    scores: dict[str, float] = defaultdict(float)
    for chunk_id, rank in vector_ranks.items():
        scores[chunk_id] += VECTOR_WEIGHT * reciprocal_rank(rank)
    for chunk_id, rank in keyword_ranks.items():
        scores[chunk_id] += KEYWORD_WEIGHT * reciprocal_rank(rank)
    for chunk_id, rank in graph_ranks.items():
        scores[chunk_id] += GRAPH_WEIGHT * reciprocal_rank(rank)
    for chunk_id, support in graph_support.items():
        scores[chunk_id] += SUPPORT_WEIGHT * support
    return dict(scores)