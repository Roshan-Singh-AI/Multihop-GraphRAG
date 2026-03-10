from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from neo4j import GraphDatabase

from .config import Settings
from .schemas import DocumentExtraction
from .utils import normalize_key


class GraphStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password.get_secret_value()),
        )

    def close(self) -> None:
        self.driver.close()

    def query(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        records, _, _ = self.driver.execute_query(
            cypher,
            parameters_=params or {},
            database_=self.settings.neo4j_database,
        )
        return [record.data() for record in records]

    def ensure_schema(self, vector_dimensions: int) -> None:
        statements = [
            "CREATE CONSTRAINT source_document_doc_id IF NOT EXISTS FOR (d:SourceDocument) REQUIRE d.doc_id IS UNIQUE",
            "CREATE CONSTRAINT chunk_chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
            "CREATE CONSTRAINT entity_key IF NOT EXISTS FOR (e:Entity) REQUIRE e.key IS UNIQUE",
            f"CREATE VECTOR INDEX {self.settings.vector_index_name} IF NOT EXISTS FOR (c:Chunk) ON (c.embedding) OPTIONS {{indexConfig: {{`vector.dimensions`: {vector_dimensions}, `vector.similarity_function`: '{self.settings.vector_similarity_function}'}}}}",
            f"CREATE FULLTEXT INDEX {self.settings.keyword_index_name} IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text, c.title]",
        ]
        for statement in statements:
            self.query(statement)

    def reset_graph(self) -> None:
        self.query("MATCH (n) DETACH DELETE n")

    def upsert_documents_and_chunks(self, chunks: list[Document], embeddings: list[list[float]]) -> None:
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            payload = {
                "doc_id": chunk.metadata["doc_id"],
                "title": chunk.metadata["title"],
                "source_path": chunk.metadata["source_path"],
                "chunk_id": chunk.metadata["chunk_id"],
                "chunk_index": chunk.metadata["chunk_index"],
                "text": chunk.page_content,
                "embedding": embedding,
            }
            self.query(
                """
                MERGE (d:SourceDocument {doc_id: $doc_id})
                SET d.title = $title,
                    d.source_path = $source_path,
                    d.updated_at = datetime()
                MERGE (c:Chunk {chunk_id: $chunk_id})
                SET c.doc_id = $doc_id,
                    c.title = $title,
                    c.source_path = $source_path,
                    c.chunk_index = $chunk_index,
                    c.text = $text,
                    c.embedding = $embedding,
                    c.updated_at = datetime()
                MERGE (d)-[:HAS_CHUNK]->(c)
                """,
                payload,
            )

    def upsert_extraction(self, chunk_id: str, extraction: DocumentExtraction) -> None:
        entities = [
            {
                "key": normalize_key(entity.name),
                "name": entity.name,
                "kind": entity.kind,
                "description": entity.description or "",
                "aliases": entity.aliases,
            }
            for entity in extraction.entities
        ]
        relationships = [
            {
                "source_key": normalize_key(relation.source),
                "source": relation.source,
                "target_key": normalize_key(relation.target),
                "target": relation.target,
                "relation": relation.relation,
                "evidence": relation.evidence or "",
                "confidence": relation.confidence,
            }
            for relation in extraction.relationships
        ]

        if entities:
            self.query(
                """
                MATCH (c:Chunk {chunk_id: $chunk_id})
                UNWIND $entities AS entity
                MERGE (e:Entity {key: entity.key})
                ON CREATE SET e.name = entity.name,
                              e.kind = entity.kind,
                              e.description = entity.description,
                              e.aliases = entity.aliases,
                              e.created_at = datetime()
                SET e.name = coalesce(e.name, entity.name),
                    e.kind = coalesce(e.kind, entity.kind),
                    e.description = CASE WHEN e.description IS NULL OR e.description = '' THEN entity.description ELSE e.description END,
                    e.aliases = CASE WHEN size(coalesce(e.aliases, [])) = 0 THEN entity.aliases ELSE e.aliases END,
                    e.updated_at = datetime()
                MERGE (c)-[:MENTIONS]->(e)
                """,
                {"chunk_id": chunk_id, "entities": entities},
            )

        if relationships:
            self.query(
                """
                MATCH (c:Chunk {chunk_id: $chunk_id})
                UNWIND $relationships AS rel
                MERGE (s:Entity {key: rel.source_key})
                ON CREATE SET s.name = rel.source, s.kind = 'concept', s.created_at = datetime()
                MERGE (t:Entity {key: rel.target_key})
                ON CREATE SET t.name = rel.target, t.kind = 'concept', t.created_at = datetime()
                MERGE (c)-[:MENTIONS]->(s)
                MERGE (c)-[:MENTIONS]->(t)
                MERGE (s)-[r:RELATES_TO {chunk_id: $chunk_id, relation: rel.relation, source_key: rel.source_key, target_key: rel.target_key}]->(t)
                SET r.evidence = rel.evidence,
                    r.confidence = rel.confidence,
                    r.updated_at = datetime()
                """,
                {"chunk_id": chunk_id, "relationships": relationships},
            )

    def vector_search(self, query_embedding: list[float], limit: int) -> list[dict[str, Any]]:
        return self.query(
            f"""
            CALL db.index.vector.queryNodes('{self.settings.vector_index_name}', $limit, $embedding)
            YIELD node, score
            RETURN node.chunk_id AS chunk_id,
                   node.doc_id AS doc_id,
                   node.title AS title,
                   node.text AS text,
                   node.source_path AS source_path,
                   score
            ORDER BY score DESC
            LIMIT $limit
            """,
            {"limit": limit, "embedding": query_embedding},
        )

    def keyword_search(self, query_text: str, limit: int) -> list[dict[str, Any]]:
        return self.query(
            f"""
            CALL db.index.fulltext.queryNodes('{self.settings.keyword_index_name}', $query)
            YIELD node, score
            RETURN node.chunk_id AS chunk_id,
                   node.doc_id AS doc_id,
                   node.title AS title,
                   node.text AS text,
                   node.source_path AS source_path,
                   score
            ORDER BY score DESC
            LIMIT $limit
            """,
            {"query": query_text, "limit": limit},
        )

    def match_entities(self, entity_names: list[str], limit: int) -> list[dict[str, Any]]:
        if not entity_names:
            return []
        return self.query(
            """
            UNWIND $entity_names AS raw
            WITH raw, toLower(raw) AS lowered, replace(replace(toLower(raw), ' ', '-'), '_', '-') AS key_guess
            MATCH (e:Entity)
            WHERE e.key = key_guess
               OR toLower(e.name) = lowered
               OR toLower(e.name) CONTAINS lowered
               OR any(alias IN coalesce(e.aliases, []) WHERE toLower(alias) = lowered)
            RETURN DISTINCT e.key AS key, e.name AS name, e.kind AS kind
            LIMIT $limit
            """,
            {"entity_names": entity_names, "limit": limit},
        )

    def seed_entities_from_chunks(self, chunk_ids: list[str], limit: int) -> list[dict[str, Any]]:
        if not chunk_ids:
            return []
        return self.query(
            """
            MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
            WHERE c.chunk_id IN $chunk_ids
            RETURN e.key AS key, e.name AS name, e.kind AS kind, count(*) AS freq
            ORDER BY freq DESC, e.name ASC
            LIMIT $limit
            """,
            {"chunk_ids": chunk_ids, "limit": limit},
        )

    def graph_search_chunks(self, entity_keys: list[str], hops: int, limit: int) -> list[dict[str, Any]]:
        if not entity_keys:
            return []
        safe_hops = max(1, min(hops, 4))
        return self.query(
            f"""
            MATCH (seed:Entity)
            WHERE seed.key IN $entity_keys
            MATCH p=(seed)-[:RELATES_TO*1..{safe_hops}]-(neighbor:Entity)
            WITH neighbor, p
            MATCH (neighbor)<-[:MENTIONS]-(chunk:Chunk)
            RETURN chunk.chunk_id AS chunk_id,
                   chunk.doc_id AS doc_id,
                   chunk.title AS title,
                   chunk.text AS text,
                   chunk.source_path AS source_path,
                   collect(DISTINCT neighbor.name)[0..5] AS supporting_entities,
                   min(length(p)) AS hop_distance,
                   count(DISTINCT p) AS path_count
            ORDER BY path_count DESC, hop_distance ASC, chunk.chunk_id ASC
            LIMIT $limit
            """,
            {"entity_keys": entity_keys, "limit": limit},
        )

    def graph_paths(self, entity_keys: list[str], hops: int, limit: int) -> list[dict[str, Any]]:
        if not entity_keys:
            return []
        safe_hops = max(1, min(hops, 4))
        return self.query(
            f"""
            MATCH (seed:Entity)
            WHERE seed.key IN $entity_keys
            MATCH p=(seed)-[:RELATES_TO*1..{safe_hops}]-(neighbor:Entity)
            RETURN [n IN nodes(p) | n.name] AS nodes,
                   [r IN relationships(p) | coalesce(r.relation, type(r))] AS relations,
                   length(p) AS hop_count
            LIMIT $limit
            """,
            {"entity_keys": entity_keys, "limit": limit},
        )

    def subgraph_for_entity(self, entity_name_or_key: str, depth: int = 2, limit: int = 40) -> dict[str, list[dict[str, Any]]]:
        lookup = self.query(
            """
            MATCH (e:Entity)
            WHERE e.key = $candidate OR toLower(e.name) = toLower($candidate)
            RETURN e.key AS key, e.name AS name
            LIMIT 1
            """,
            {"candidate": normalize_key(entity_name_or_key)},
        )
        if not lookup:
            lookup = self.query(
                """
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS toLower($candidate)
                RETURN e.key AS key, e.name AS name
                LIMIT 1
                """,
                {"candidate": entity_name_or_key},
            )
        if not lookup:
            return {"nodes": [], "edges": []}
        key = lookup[0]["key"]
        safe_depth = max(1, min(depth, 3))
        node_rows = self.query(
            f"""
            MATCH (seed:Entity {{key: $key}})
            OPTIONAL MATCH p=(seed)-[:RELATES_TO*1..{safe_depth}]-(neighbor:Entity)
            WITH collect(DISTINCT seed) + collect(DISTINCT neighbor) AS nodes
            UNWIND nodes AS n
            WITH DISTINCT n
            RETURN n.key AS id, n.name AS label, n.kind AS kind
            LIMIT $limit
            """,
            {"key": key, "limit": limit},
        )
        edge_rows = self.query(
            f"""
            MATCH (seed:Entity {{key: $key}})
            MATCH p=(seed)-[rels:RELATES_TO*1..{safe_depth}]-(neighbor:Entity)
            UNWIND rels AS r
            RETURN DISTINCT startNode(r).key AS source,
                            endNode(r).key AS target,
                            coalesce(r.relation, type(r)) AS relation
            LIMIT $limit
            """,
            {"key": key, "limit": limit},
        )
        return {"nodes": node_rows, "edges": edge_rows}
