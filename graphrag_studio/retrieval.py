from __future__ import annotations

from dataclasses import dataclass

from langchain_huggingface import HuggingFaceEmbeddings

from .config import Settings
from .extractor import GraphExtractor
from .graph_store import GraphStore
from .ranking import fuse_candidate_scores
from .schemas import GraphPath, HybridRetrievalResult, RetrievedChunk


@dataclass(slots=True)
class HybridRetriever:
    settings: Settings
    graph_store: GraphStore
    extractor: GraphExtractor
    embeddings: HuggingFaceEmbeddings

    def retrieve_vector_only(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
        limit = top_k or self.settings.retrieval_top_k
        query_vector = self.embeddings.embed_query(question)
        rows = self.graph_store.vector_search(query_vector, limit)
        chunks: list[RetrievedChunk] = []
        for rank, row in enumerate(rows, start=1):
            chunks.append(
                RetrievedChunk(
                    chunk_id=row["chunk_id"],
                    doc_id=row["doc_id"],
                    title=row["title"],
                    text=row["text"],
                    source_path=row["source_path"],
                    vector_rank=rank,
                    vector_score=float(row["score"]),
                    final_score=float(row["score"]),
                )
            )
        return chunks

    def retrieve(self, question: str, top_k: int | None = None, hops: int | None = None) -> HybridRetrievalResult:
        limit = top_k or self.settings.retrieval_top_k
        hops = hops or self.settings.graph_max_hops

        query_vector = self.embeddings.embed_query(question)
        vector_rows = self.graph_store.vector_search(query_vector, limit * 3)
        keyword_rows = self.graph_store.keyword_search(question, limit * 2)

        query_entities = self.extractor.extract_query_entities(question)
        matched_entities = self.graph_store.match_entities(query_entities, limit=self.settings.query_seed_entity_limit)
        seed_from_vector = self.graph_store.seed_entities_from_chunks(
            [row["chunk_id"] for row in vector_rows[: min(4, len(vector_rows))]],
            limit=self.settings.chunk_entity_limit,
        )

        entity_seed_map: dict[str, str] = {}
        for row in matched_entities + seed_from_vector:
            entity_seed_map[row["key"]] = row["name"]
        entity_keys = list(entity_seed_map.keys())

        graph_rows = self.graph_store.graph_search_chunks(entity_keys, hops=hops, limit=limit * 3)
        path_rows = self.graph_store.graph_paths(entity_keys, hops=hops, limit=limit * 2)

        candidates: dict[str, RetrievedChunk] = {}
        vector_ranks: dict[str, int] = {}
        keyword_ranks: dict[str, int] = {}
        graph_ranks: dict[str, int] = {}
        graph_support: dict[str, float] = {}

        for rank, row in enumerate(vector_rows, start=1):
            vector_ranks[row["chunk_id"]] = rank
            candidates.setdefault(
                row["chunk_id"],
                RetrievedChunk(
                    chunk_id=row["chunk_id"],
                    doc_id=row["doc_id"],
                    title=row["title"],
                    text=row["text"],
                    source_path=row["source_path"],
                ),
            )
            candidates[row["chunk_id"]].vector_rank = rank
            candidates[row["chunk_id"]].vector_score = float(row["score"])

        for rank, row in enumerate(keyword_rows, start=1):
            keyword_ranks[row["chunk_id"]] = rank
            candidates.setdefault(
                row["chunk_id"],
                RetrievedChunk(
                    chunk_id=row["chunk_id"],
                    doc_id=row["doc_id"],
                    title=row["title"],
                    text=row["text"],
                    source_path=row["source_path"],
                ),
            )
            candidates[row["chunk_id"]].keyword_rank = rank
            candidates[row["chunk_id"]].keyword_score = float(row["score"])

        max_path_count = max((int(row["path_count"]) for row in graph_rows), default=1)
        for rank, row in enumerate(graph_rows, start=1):
            graph_ranks[row["chunk_id"]] = rank
            support = 0.6 * (int(row["path_count"]) / max_path_count) + 0.4 * (1.0 / (int(row["hop_distance"]) + 1))
            graph_support[row["chunk_id"]] = support
            candidates.setdefault(
                row["chunk_id"],
                RetrievedChunk(
                    chunk_id=row["chunk_id"],
                    doc_id=row["doc_id"],
                    title=row["title"],
                    text=row["text"],
                    source_path=row["source_path"],
                ),
            )
            candidates[row["chunk_id"]].graph_rank = rank
            candidates[row["chunk_id"]].graph_score = support
            candidates[row["chunk_id"]].supporting_entities = list(row.get("supporting_entities", []))

        fused_scores = fuse_candidate_scores(vector_ranks, keyword_ranks, graph_ranks, graph_support)
        for chunk_id, candidate in candidates.items():
            candidate.final_score = fused_scores.get(chunk_id, 0.0)

        ranked_chunks = sorted(candidates.values(), key=lambda chunk: chunk.final_score, reverse=True)[:limit]
        graph_paths = [
            GraphPath(nodes=row["nodes"], relations=row["relations"], hop_count=int(row["hop_count"]))
            for row in path_rows
        ]
        return HybridRetrievalResult(
            linked_entities=list(entity_seed_map.values()),
            chunks=ranked_chunks,
            paths=graph_paths,
            debug={
                "query_entities": query_entities,
                "seed_entity_keys": entity_keys,
                "vector_candidates": len(vector_rows),
                "keyword_candidates": len(keyword_rows),
                "graph_candidates": len(graph_rows),
            },
        )
