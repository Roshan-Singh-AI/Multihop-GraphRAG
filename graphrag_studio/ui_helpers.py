from __future__ import annotations

import math
from typing import Any

import networkx as nx
import pandas as pd
import plotly.graph_objects as go

from .schemas import AnswerPacket, BenchmarkSummary



def chunks_to_dataframe(packet: AnswerPacket) -> pd.DataFrame:
    rows = []
    for chunk in packet.retrieval.chunks:
        rows.append(
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "title": chunk.title,
                "vector_rank": chunk.vector_rank,
                "keyword_rank": chunk.keyword_rank,
                "graph_rank": chunk.graph_rank,
                "vector_score": round(chunk.vector_score, 4),
                "keyword_score": round(chunk.keyword_score, 4),
                "graph_score": round(chunk.graph_score, 4),
                "final_score": round(chunk.final_score, 4),
                "supporting_entities": ", ".join(chunk.supporting_entities),
            }
        )
    return pd.DataFrame(rows)



def benchmark_to_dataframe(summary: BenchmarkSummary) -> pd.DataFrame:
    rows = [row.model_dump() for row in summary.rows]
    return pd.DataFrame(rows)



def build_subgraph_figure(subgraph: dict[str, list[dict[str, Any]]]) -> go.Figure:
    graph = nx.Graph()
    for node in subgraph.get("nodes", []):
        graph.add_node(node["id"], label=node["label"], kind=node.get("kind", "entity"))
    for edge in subgraph.get("edges", []):
        graph.add_edge(edge["source"], edge["target"], relation=edge.get("relation", "RELATES_TO"))

    figure = go.Figure()
    if graph.number_of_nodes() == 0:
        figure.update_layout(title="No graph data available yet")
        return figure

    layout = nx.spring_layout(graph, seed=7, k=max(0.35, 1.4 / math.sqrt(max(1, graph.number_of_nodes()))))

    edge_x: list[float] = []
    edge_y: list[float] = []
    for source, target in graph.edges():
        x0, y0 = layout[source]
        x1, y1 = layout[target]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    figure.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line={"width": 1},
            hoverinfo="skip",
            showlegend=False,
        )
    )

    node_x = []
    node_y = []
    node_text = []
    for node_id, attrs in graph.nodes(data=True):
        x, y = layout[node_id]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{attrs['label']}\n({attrs.get('kind', 'entity')})")

    figure.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=[graph.nodes[node_id]["label"] for node_id in graph.nodes()],
            textposition="top center",
            hovertext=node_text,
            hoverinfo="text",
            marker={"size": 18},
            showlegend=False,
        )
    )

    figure.update_layout(
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        xaxis={"visible": False},
        yaxis={"visible": False},
        title="Entity neighborhood",
        height=520,
    )
    return figure



def build_benchmark_figure(summary: BenchmarkSummary) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=["Vector-only hit rate", "Hybrid hit rate"],
            y=[summary.vector_hit_rate, summary.hybrid_hit_rate],
            text=[f"{summary.vector_hit_rate:.0%}", f"{summary.hybrid_hit_rate:.0%}"],
            textposition="outside",
        )
    )
    figure.update_layout(height=360, margin={"l": 20, "r": 20, "t": 30, "b": 20})
    return figure
