from __future__ import annotations

import argparse

from graphrag_studio.service import GraphRAGService


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the bundled GraphRAG benchmark")
    parser.add_argument("--cases", default="data/benchmark/heldout_queries.json")
    args = parser.parse_args()

    service = GraphRAGService()
    summary = service.run_benchmark(args.cases)
    print(summary.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
