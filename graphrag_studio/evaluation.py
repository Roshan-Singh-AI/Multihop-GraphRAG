from __future__ import annotations

import json
from pathlib import Path

from .ranking import hit_at_k, reciprocal_rank_for_docs
from .schemas import BenchmarkCase, BenchmarkRow, BenchmarkSummary



def load_benchmark_cases(path: Path) -> list[BenchmarkCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [BenchmarkCase.model_validate(item) for item in payload]



def run_benchmark(cases: list[BenchmarkCase], service: object, top_k: int = 5) -> BenchmarkSummary:
    rows: list[BenchmarkRow] = []
    vector_hits = 0
    hybrid_hits = 0
    vector_mrr_total = 0.0
    hybrid_mrr_total = 0.0

    for case in cases:
        vector_chunks = service.retriever.retrieve_vector_only(case.question, top_k=top_k)
        hybrid_result = service.retriever.retrieve(case.question, top_k=top_k, hops=service.settings.graph_max_hops)

        expected_doc_ids = set(case.expected_doc_ids)
        vector_doc_ids = [chunk.doc_id for chunk in vector_chunks]
        hybrid_doc_ids = [chunk.doc_id for chunk in hybrid_result.chunks]

        vector_hit = hit_at_k(vector_doc_ids, expected_doc_ids)
        hybrid_hit = hit_at_k(hybrid_doc_ids, expected_doc_ids)
        vector_mrr = reciprocal_rank_for_docs(vector_doc_ids, expected_doc_ids)
        hybrid_mrr = reciprocal_rank_for_docs(hybrid_doc_ids, expected_doc_ids)

        vector_hits += int(vector_hit)
        hybrid_hits += int(hybrid_hit)
        vector_mrr_total += vector_mrr
        hybrid_mrr_total += hybrid_mrr

        rows.append(
            BenchmarkRow(
                case_id=case.case_id,
                question=case.question,
                vector_hit=vector_hit,
                hybrid_hit=hybrid_hit,
                vector_mrr=vector_mrr,
                hybrid_mrr=hybrid_mrr,
            )
        )

    total = max(1, len(cases))
    vector_hit_rate = vector_hits / total
    hybrid_hit_rate = hybrid_hits / total
    relative_improvement_pct = 0.0
    if vector_hit_rate > 0:
        relative_improvement_pct = ((hybrid_hit_rate - vector_hit_rate) / vector_hit_rate) * 100.0

    return BenchmarkSummary(
        total_cases=len(cases),
        vector_hit_rate=vector_hit_rate,
        hybrid_hit_rate=hybrid_hit_rate,
        relative_improvement_pct=relative_improvement_pct,
        vector_mrr=vector_mrr_total / total,
        hybrid_mrr=hybrid_mrr_total / total,
        rows=rows,
    )
