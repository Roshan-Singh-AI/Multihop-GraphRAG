from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from graphrag_studio.service import GraphRAGService
from graphrag_studio.ui_helpers import (
    benchmark_to_dataframe,
    build_benchmark_figure,
    build_subgraph_figure,
    chunks_to_dataframe,
)

st.set_page_config(page_title="GraphRAG Studio", page_icon="🕸️", layout="wide")


@st.cache_resource(show_spinner=False)
def get_service() -> GraphRAGService:
    return GraphRAGService()


service = get_service()

if "last_answer" not in st.session_state:
    st.session_state["last_answer"] = None
if "last_benchmark" not in st.session_state:
    st.session_state["last_benchmark"] = None

st.markdown(
    """
    <style>
      .hero {
        padding: 1.2rem 1.4rem;
        border-radius: 1rem;
        background: linear-gradient(135deg, rgba(99,102,241,.12), rgba(16,185,129,.10));
        border: 1px solid rgba(148,163,184,.25);
        margin-bottom: 1rem;
      }
      .subtle {
        color: rgba(100,116,139,1);
      }
    </style>
    <div class="hero">
      <h1 style="margin:0;">GraphRAG Studio</h1>
      <p class="subtle" style="margin: .35rem 0 0 0;">
        Context-aware Knowledge Graph RAG with Groq-powered ETL, Neo4j traversal, and a retrieval inspector built for portfolio demos.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Control Plane")
    health = service.health()
    st.write("**Groq configured**", "Yes" if health["groq_configured"] else "No")
    st.write("**Neo4j**", health["neo4j_uri"])
    st.write("**Embedding model**", health["embedding_model"])

    if st.button("Ingest bundled sample corpus", use_container_width=True):
        with st.spinner("Ingesting sample corpus into Neo4j..."):
            result = service.ingest_path("data/sample_corpus")
        st.success(
            f"Loaded {result.document_count} docs, {result.chunk_count} chunks, {result.entity_count} entities, {result.relation_count} relations."
        )

    uploads = st.file_uploader(
        "Upload .md / .txt / .pdf / .json docs",
        type=["md", "txt", "pdf", "json"],
        accept_multiple_files=True,
    )
    if st.button("Ingest uploaded files", disabled=not uploads, use_container_width=True):
        with tempfile.TemporaryDirectory() as tmp_dir:
            upload_root = Path(tmp_dir)
            for uploaded in uploads or []:
                target = upload_root / uploaded.name
                target.write_bytes(uploaded.getbuffer())
            with st.spinner("Parsing and ingesting uploaded files..."):
                result = service.ingest_path(upload_root)
        st.success(
            f"Loaded {result.document_count} docs, {result.chunk_count} chunks, {result.entity_count} entities, {result.relation_count} relations."
        )

    if st.button("Reset graph", use_container_width=True):
        service.reset_graph()
        st.warning("Neo4j graph cleared.")

example_questions = [
    "Which plants are affected if Orion Gateway firmware issue KB-214 impacts cameras used for weld quality?",
    "Which team should coordinate with compliance if Atlas Plant pauses WeldVision inspections?",
    "How does Mercury Data Lake feed Helios Platform analytics for weld defects?",
]

ask_tab, graph_tab, trace_tab, benchmark_tab, architecture_tab = st.tabs(
    ["Ask", "Graph Explorer", "Retrieval Inspector", "Benchmark", "Architecture"]
)

with ask_tab:
    st.subheader("Run a multi-hop question")
    selected_example = st.selectbox("Try a demo query", [""] + example_questions)
    question = st.text_input(
        "Question",
        value=selected_example,
        placeholder="Ask about plants, teams, standards, incidents, suppliers, or platform dependencies...",
    )
    if st.button("Run GraphRAG", use_container_width=True, type="primary") and question.strip():
        with st.spinner("Retrieving vector evidence, expanding graph paths, and generating answer..."):
            st.session_state["last_answer"] = service.answer(question)

    packet = st.session_state.get("last_answer")
    if packet is not None:
        st.markdown("### Answer")
        st.write(packet.answer)
        stats_col1, stats_col2, stats_col3 = st.columns(3)
        stats_col1.metric("Retrieved chunks", len(packet.retrieval.chunks))
        stats_col2.metric("Linked entities", len(packet.retrieval.linked_entities))
        stats_col3.metric("Graph paths", len(packet.retrieval.paths))

        if packet.citations:
            st.caption("Top citations: " + ", ".join(packet.citations))

        with st.expander("Retrieved evidence"):
            for chunk in packet.retrieval.chunks:
                st.markdown(f"**{chunk.chunk_id} — {chunk.title}**")
                st.write(chunk.text)
                if chunk.supporting_entities:
                    st.caption("Supporting entities: " + ", ".join(chunk.supporting_entities))
                st.divider()

with graph_tab:
    st.subheader("Inspect the entity neighborhood")
    default_entity = ""
    packet = st.session_state.get("last_answer")
    if packet and packet.retrieval.linked_entities:
        default_entity = packet.retrieval.linked_entities[0]
    entity_name = st.text_input("Entity", value=default_entity, placeholder="Try Orion Gateway or Atlas Plant")
    depth = st.slider("Traversal depth", min_value=1, max_value=3, value=2)
    if entity_name:
        subgraph = service.subgraph(entity_name, depth=depth)
        st.plotly_chart(build_subgraph_figure(subgraph), use_container_width=True)
        if not subgraph["nodes"]:
            st.info("Ingest the sample corpus first, then explore an entity name from the demo documents.")

with trace_tab:
    st.subheader("Why a chunk ranked where it did")
    packet = st.session_state.get("last_answer")
    if packet is None:
        st.info("Run a question first. This tab will show vector, keyword, and graph contributions for each chunk.")
    else:
        st.dataframe(chunks_to_dataframe(packet), use_container_width=True, hide_index=True)
        if packet.retrieval.paths:
            st.markdown("### Graph paths")
            for path in packet.retrieval.paths:
                st.markdown(f"- {' → '.join(path.nodes)}  ")
                st.caption("Relations: " + " | ".join(path.relations))

with benchmark_tab:
    st.subheader("Compare hybrid retrieval against vector-only")
    if st.button("Run bundled benchmark", use_container_width=True):
        with st.spinner("Running held-out retrieval benchmark..."):
            st.session_state["last_benchmark"] = service.run_benchmark("data/benchmark/heldout_queries.json")
    summary = st.session_state.get("last_benchmark")
    if summary is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Vector hit rate", f"{summary.vector_hit_rate:.0%}")
        c2.metric("Hybrid hit rate", f"{summary.hybrid_hit_rate:.0%}")
        c3.metric("Relative improvement", f"{summary.relative_improvement_pct:.1f}%")
        st.plotly_chart(build_benchmark_figure(summary), use_container_width=True)
        st.dataframe(benchmark_to_dataframe(summary), use_container_width=True, hide_index=True)
    else:
        st.info("The benchmark uses the bundled sample corpus and compares vector-only retrieval to hybrid GraphRAG retrieval.")

with architecture_tab:
    st.subheader("Repository notes")
    architecture_doc = Path("docs/architecture.md")
    if architecture_doc.exists():
        st.markdown(architecture_doc.read_text(encoding="utf-8"))
    else:
        st.info("Architecture notes are not available.")
