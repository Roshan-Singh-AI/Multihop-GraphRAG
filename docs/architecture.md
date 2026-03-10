# Architecture Notes

## Design goal

The system is built to answer questions that require **more than one retrieval hop**.

A vector-only retriever can find semantically similar chunks, but it often misses linked evidence chains such as:

- system -> component -> plant
- inspection system -> quality standard -> compliance owner
- incident -> runbook -> responsible team

GraphRAG Studio addresses that gap by combining three evidence channels:

1. **Vector retrieval** for semantic recall
2. **Keyword retrieval** for exact terms like KB numbers or standard names
3. **Graph traversal** for multi-hop expansion over extracted entities and relationships

## Data model

### Nodes

- `SourceDocument`
- `Chunk`
- `Entity`

### Relationships

- `(:SourceDocument)-[:HAS_CHUNK]->(:Chunk)`
- `(:Chunk)-[:MENTIONS]->(:Entity)`
- `(:Entity)-[:RELATES_TO {relation, evidence, confidence}]->(:Entity)`

## Ingestion flow

1. Parse `.md`, `.txt`, `.pdf`, and `.json` documents.
2. Split documents into chunks.
3. Generate embeddings for each chunk.
4. Store chunks in Neo4j and build a vector index.
5. Run Groq structured extraction on each chunk.
6. Upsert entities and relations into the graph.

## Retrieval flow

1. Embed the question.
2. Run vector search over chunk embeddings.
3. Run keyword search over chunk text and titles.
4. Extract query entities.
5. Match those entities against graph entities.
6. Expand graph paths up to `N` hops.
7. Pull graph-supported chunks connected to expanded entities.
8. Fuse vector, keyword, and graph signals.
9. Send the top evidence into the answer generation step.

## Why the graph matters

The graph helps when the question asks for:

- affected scope
- ownership
- dependencies
- policy / compliance implications
- impact propagation

Those are usually relationship-heavy questions, not pure semantic similarity questions.
