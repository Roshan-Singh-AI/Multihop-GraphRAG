from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ExtractedEntity(BaseModel):
    name: str
    kind: str = "concept"
    description: str | None = None
    aliases: list[str] = Field(default_factory=list)


class ExtractedRelation(BaseModel):
    source: str
    target: str
    relation: str
    evidence: str | None = None
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)


class DocumentExtraction(BaseModel):
    summary: str = ""
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relationships: list[ExtractedRelation] = Field(default_factory=list)
    salience_tags: list[str] = Field(default_factory=list)


class QueryEntitySet(BaseModel):
    entities: list[str] = Field(default_factory=list)


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    text: str
    source_path: str
    vector_rank: int | None = None
    keyword_rank: int | None = None
    graph_rank: int | None = None
    vector_score: float = 0.0
    keyword_score: float = 0.0
    graph_score: float = 0.0
    final_score: float = 0.0
    supporting_entities: list[str] = Field(default_factory=list)


class GraphPath(BaseModel):
    nodes: list[str]
    relations: list[str]
    hop_count: int
    supporting_chunk_ids: list[str] = Field(default_factory=list)


class HybridRetrievalResult(BaseModel):
    linked_entities: list[str] = Field(default_factory=list)
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    paths: list[GraphPath] = Field(default_factory=list)
    debug: dict[str, Any] = Field(default_factory=dict)


class AnswerPacket(BaseModel):
    question: str
    answer: str
    retrieval: HybridRetrievalResult
    citations: list[str] = Field(default_factory=list)


class IngestResult(BaseModel):
    document_count: int
    chunk_count: int
    entity_count: int
    relation_count: int
    sample_titles: list[str] = Field(default_factory=list)


class BenchmarkCase(BaseModel):
    case_id: str
    question: str
    expected_doc_ids: list[str]
    expected_entities: list[str] = Field(default_factory=list)
    notes: str | None = None


class BenchmarkRow(BaseModel):
    case_id: str
    question: str
    vector_hit: bool
    hybrid_hit: bool
    vector_mrr: float
    hybrid_mrr: float


class BenchmarkSummary(BaseModel):
    total_cases: int
    vector_hit_rate: float
    hybrid_hit_rate: float
    relative_improvement_pct: float
    vector_mrr: float
    hybrid_mrr: float
    rows: list[BenchmarkRow] = Field(default_factory=list)


class QueryRequest(BaseModel):
    question: str
    top_k: int = 6
    hops: int = 2


class PathIngestRequest(BaseModel):
    path: str


class SubgraphResponse(BaseModel):
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
