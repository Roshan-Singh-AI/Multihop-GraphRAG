from __future__ import annotations

from graphrag_studio.service import GraphRAGService

if __name__ == "__main__":
    service = GraphRAGService()
    result = service.ingest_path("data/sample_corpus")
    print(result.model_dump_json(indent=2))
