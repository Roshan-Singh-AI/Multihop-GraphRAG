# Showcase Notes

## Suggested portfolio pitch

I built a context-aware GraphRAG system that combines semantic retrieval with Neo4j traversal for multi-hop reasoning. It uses Groq structured outputs to automate entity and relationship extraction, then benchmarks hybrid retrieval against a vector-only baseline.

## Suggested README callout

- hybrid retrieval over vector-only RAG
- graph ETL with structured outputs
- retrieval inspector for explainability
- benchmark harness for measurable improvement

## Suggested screenshot set

1. Ask tab with a multi-hop answer and citations
2. Graph Explorer centered on `Orion Gateway`
3. Retrieval Inspector table with score breakdowns
4. Benchmark tab showing hybrid retrieval outperforming vector-only

## Suggested interview talking points

- why vector-only retrieval breaks on dependency questions
- why structured outputs simplify graph ETL reliability
- how graph traversal improves scope/ownership/compliance questions
- what you would change when moving from 10 demo docs to 5K+ real docs

## Good public framing

Say this repo is a **public architecture + demo implementation** of the pattern you used, while keeping any private enterprise corpus and proprietary benchmark data separate.
