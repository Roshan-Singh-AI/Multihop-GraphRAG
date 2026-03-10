# Benchmark Notes

The included benchmark is intentionally simple: it measures retrieval quality against a held-out set of known target documents.

## Compared systems

### Vector-only baseline
Uses the chunk embedding index only.

### Hybrid GraphRAG retriever
Uses:

- vector retrieval
- keyword retrieval
- graph traversal
- score fusion

## Metrics

- **Hit rate**: did any expected document appear in the top-k?
- **MRR**: how early did the first relevant document appear?
- **Relative improvement %**: hit-rate improvement over vector-only

## Why this matters in a portfolio repo

It turns the project into an engineering artifact with a measurable retrieval story instead of a purely architectural story.
