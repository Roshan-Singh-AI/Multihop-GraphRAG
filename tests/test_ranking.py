from graphrag_studio.ranking import fuse_candidate_scores, reciprocal_rank, reciprocal_rank_for_docs


def test_reciprocal_rank_monotonic() -> None:
    assert reciprocal_rank(1) > reciprocal_rank(2) > reciprocal_rank(3)



def test_reciprocal_rank_for_docs() -> None:
    score = reciprocal_rank_for_docs(["a", "b", "c"], {"b"})
    assert score == 0.5



def test_fuse_candidate_scores_prefers_multi_signal_chunks() -> None:
    scores = fuse_candidate_scores(
        vector_ranks={"chunk_a": 1, "chunk_b": 2},
        keyword_ranks={"chunk_b": 1},
        graph_ranks={"chunk_b": 1},
        graph_support={"chunk_b": 1.0},
    )
    assert scores["chunk_b"] > scores["chunk_a"]



def test_hit_at_k_with_match() -> None:
    """Test hit_at_k returns True when expected doc is found."""
    doc_ids = ["doc-1", "doc-2", "doc-3"]
    expected = {"doc-2", "doc-5"}
    assert hit_at_k(doc_ids, expected) is True



def test_reciprocal_rank_edge_cases() -> None:
    """Test reciprocal_rank handles edge cases correctly."""
    assert reciprocal_rank(None) == 0.0
    assert reciprocal_rank(0) == 0.0
    assert reciprocal_rank(-1) == 0.0
    assert reciprocal_rank(1) > 0.0
