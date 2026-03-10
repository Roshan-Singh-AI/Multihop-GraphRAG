from __future__ import annotations

from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from .chunking import chunk_documents
from .config import Settings, get_settings
from .embeddings import build_embeddings
from .evaluation import load_benchmark_cases, run_benchmark
from .extractor import GraphExtractor
from .graph_store import GraphStore
from .loaders import load_documents
from .retrieval import HybridRetriever
from .schemas import AnswerPacket, BenchmarkSummary, HybridRetrievalResult, IngestResult
from .utils import truncate_text


class GraphRAGService:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.embeddings = build_embeddings(self.settings)
        self.graph_store = GraphStore(self.settings)
        self.extractor = GraphExtractor(self.settings)
        self.retriever = HybridRetriever(self.settings, self.graph_store, self.extractor, self.embeddings)
        self._answer_chain = None

    def health(self) -> dict[str, object]:
        return {
            "app": self.settings.app_name,
            "groq_configured": self.settings.groq_api_key is not None,
            "neo4j_uri": self.settings.neo4j_uri,
            "embedding_model": self.settings.embedding_model_name,
        }

    def reset_graph(self) -> None:
        self.graph_store.reset_graph()

    def ingest_path(self, path: str | Path) -> IngestResult:
        path = Path(path)
        documents = load_documents(path)
        chunks = chunk_documents(documents, self.settings)
        embeddings = self.embeddings.embed_documents([chunk.page_content for chunk in chunks])
        vector_dimensions = len(embeddings[0]) if embeddings else 384
        self.graph_store.ensure_schema(vector_dimensions)
        self.graph_store.upsert_documents_and_chunks(chunks, embeddings)

        entity_count = 0
        relation_count = 0
        for chunk in chunks:
            extraction = self.extractor.extract_chunk(
                chunk.page_content,
                title=chunk.metadata["title"],
                source_path=chunk.metadata["source_path"],
            )
            entity_count += len(extraction.entities)
            relation_count += len(extraction.relationships)
            self.graph_store.upsert_extraction(chunk.metadata["chunk_id"], extraction)

        return IngestResult(
            document_count=len(documents),
            chunk_count=len(chunks),
            entity_count=entity_count,
            relation_count=relation_count,
            sample_titles=[document.title for document in documents[:5]],
        )

    def answer(self, question: str, top_k: int | None = None, hops: int | None = None) -> AnswerPacket:
        retrieval = self.retriever.retrieve(question, top_k=top_k, hops=hops)
        answer = self._generate_answer(question, retrieval)
        citations = [chunk.chunk_id for chunk in retrieval.chunks[:5]]
        return AnswerPacket(question=question, answer=answer, retrieval=retrieval, citations=citations)

    def subgraph(self, entity_name_or_key: str, depth: int = 2) -> dict[str, list[dict[str, object]]]:
        return self.graph_store.subgraph_for_entity(entity_name_or_key, depth=depth)

    def run_benchmark(self, cases_path: str | Path) -> BenchmarkSummary:
        cases = load_benchmark_cases(Path(cases_path))
        return run_benchmark(cases, self)

    def _generate_answer(self, question: str, retrieval: HybridRetrievalResult) -> str:
        if not retrieval.chunks:
            return "No evidence was retrieved. Ingest documents first, or try a more specific question."

        if self.settings.groq_api_key is None:
            evidence_lines = [
                f"- {chunk.title}: {truncate_text(chunk.text, 180)} [{chunk.chunk_id}]"
                for chunk in retrieval.chunks[:4]
            ]
            return (
                "Groq generation is not configured, so this is a retrieval-only summary. The strongest evidence suggests:\n"
                + "\n".join(evidence_lines)
            )

        if self._answer_chain is None:
            llm = ChatGroq(
                model=self.settings.groq_answer_model,
                temperature=0.1,
                max_retries=self.settings.groq_max_retries,
            )
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are GraphRAG Studio, an enterprise research assistant. Answer only from the supplied evidence. "
                        "Use inline chunk citations like [doc::chunk_000]. If evidence is missing, say so clearly.",
                    ),
                    (
                        "human",
                        "Question: {question}\n\nRetrieved chunks:\n{context}\n\nGraph evidence:\n{paths}\n\n"
                        "Write a direct answer, then a short note on why the graph traversal mattered, and finish with sources.",
                    ),
                ]
            )
            self._answer_chain = prompt | llm | StrOutputParser()

        context = "\n\n".join(
            f"[{chunk.chunk_id}] {chunk.text}" for chunk in retrieval.chunks[: min(5, len(retrieval.chunks))]
        )
        paths = "\n".join(
            f"- {' -> '.join(path.nodes)} via {' | '.join(path.relations)}"
            for path in retrieval.paths[: min(5, len(retrieval.paths))]
        ) or "- No graph path expansion available"
        return self._answer_chain.invoke({"question": question, "context": context, "paths": paths})
