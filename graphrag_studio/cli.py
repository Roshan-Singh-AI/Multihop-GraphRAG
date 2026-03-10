from __future__ import annotations

import json
from pathlib import Path

import typer

from .service import GraphRAGService

app = typer.Typer(no_args_is_help=True)


@app.command()
def ingest(path: Path) -> None:
    service = GraphRAGService()
    result = service.ingest_path(path)
    typer.echo(json.dumps(result.model_dump(), indent=2))


@app.command()
def ask(question: str) -> None:
    service = GraphRAGService()
    packet = service.answer(question)
    typer.echo(packet.answer)


@app.command()
def benchmark(cases: Path = Path("data/benchmark/heldout_queries.json")) -> None:
    service = GraphRAGService()
    summary = service.run_benchmark(cases)
    typer.echo(json.dumps(summary.model_dump(), indent=2))


@app.command()
def reset() -> None:
    service = GraphRAGService()
    service.reset_graph()
    typer.echo("Graph reset completed.")


if __name__ == "__main__":
    app()
