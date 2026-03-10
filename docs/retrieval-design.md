# Retrieval Design

## Hybrid retrieval recipe

The retriever deliberately combines three ranking signals.

### 1. Vector signal
Used for broad semantic recall.

### 2. Keyword signal
Useful for exact identifiers such as:

- `KB-214`
- `QA-17`
- product / system names

### 3. Graph signal
Generated from multi-hop traversal around linked entities. This is what makes the system context-aware.

## Ranking strategy

Chunks are fused using weighted reciprocal rank plus a graph support bonus.

```text
final_score =
    0.55 * rrf(vector_rank)
  + 0.15 * rrf(keyword_rank)
  + 0.20 * rrf(graph_rank)
  + 0.10 * graph_support
```

Where `graph_support` rewards:

- more supporting paths
- shorter hop distance

## Query seeding strategy

Seed entities come from two places:

1. entities extracted directly from the user question
2. entities mentioned by the strongest vector-retrieved chunks

That gives the retriever a useful fallback when the user asks an implicit question without naming every important node explicitly.

## Failure mode handled better than vector-only

Question:

> Which plants are affected if Orion Gateway firmware issue KB-214 impacts cameras used for weld quality?

A vector-only system may retrieve the firmware notice and maybe the camera guide.

The graph-aware system can connect:

- `Orion Gateway`
- `WeldVision Camera`
- `Atlas Plant`
- `Nova Plant`

and return the impacted plants directly.
