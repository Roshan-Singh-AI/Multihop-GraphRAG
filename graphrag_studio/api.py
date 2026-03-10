from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI

from .config import get_settings
from .schemas import AnswerPacket, BenchmarkSummary, PathIngestRequest, QueryRequest, SubgraphResponse
from .service import GraphRAGService

app = FastAPI(title="GraphRAG Studio API", version="0.1.0")


@lru_cache(maxsize=1)
def get_service() -> GraphRAGService:
    return GraphRAGService(get_settings())


@app.get("/health")
def health() -> dict[str, object]:
    return get_service().health()


@app.post("/ingest/path")
def ingest_path(request: PathIngestRequest) -> dict[str, object]:
    result = get_service().ingest_path(request.path)
    return result.model_dump()


@app.post("/query", response_model=AnswerPacket)
def query(request: QueryRequest) -> AnswerPacket:
    return get_service().answer(request.question, top_k=request.top_k, hops=request.hops)


@app.post("/benchmark", response_model=BenchmarkSummary)
def benchmark(request: PathIngestRequest) -> BenchmarkSummary:
    return get_service().run_benchmark(request.path)


@app.get("/subgraph", response_model=SubgraphResponse)
def subgraph(entity: str, depth: int = 2) -> SubgraphResponse:
    return SubgraphResponse(**get_service().subgraph(entity, depth=depth))
