from __future__ import annotations

from typing import TypeVar

from groq import Groq
from pydantic import BaseModel

from .config import Settings
from .schemas import DocumentExtraction, ExtractedEntity, ExtractedRelation, QueryEntitySet
from .utils import heuristic_entity_candidates, normalize_key, sentence_split, truncate_text

ModelT = TypeVar("ModelT", bound=BaseModel)

RELATION_KEYWORDS = {
    "depends on": "DEPENDS_ON",
    "uses": "USES",
    "connects to": "CONNECTS_TO",
    "feeds": "FEEDS",
    "owned by": "OWNED_BY",
    "managed by": "MANAGED_BY",
    "provided by": "PROVIDED_BY",
    "supplied by": "SUPPLIED_BY",
    "deployed at": "DEPLOYED_AT",
    "runs at": "RUNS_AT",
    "requires": "REQUIRES",
    "governed by": "GOVERNED_BY",
    "covered by": "COVERED_BY",
    "reports to": "REPORTS_TO",
    "coordinates with": "COORDINATES_WITH",
    "routes to": "ROUTES_TO",
}


class GraphExtractor:
    def __init__(self, settings: Settings):
        self.settings = settings
        api_key = settings.groq_api_key.get_secret_value() if settings.groq_api_key else None
        self.client = Groq(api_key=api_key) if api_key else None

    def extract_chunk(self, text: str, *, title: str, source_path: str) -> DocumentExtraction:
        if self.client is not None:
            try:
                return self._llm_extract_chunk(text=text, title=title, source_path=source_path)
            except Exception:
                pass
        return self._heuristic_extract_chunk(text)

    def extract_query_entities(self, question: str) -> list[str]:
        if self.client is not None:
            try:
                result = self._structured_call(
                    schema=QueryEntitySet,
                    system_prompt=(
                        "Extract the minimal set of canonical entity names needed to answer the question. "
                        "Prefer products, teams, systems, plants, suppliers, incidents, and standards."
                    ),
                    user_payload=question,
                    model=self.settings.groq_extraction_model,
                )
                return [entity for entity in result.entities if entity]
            except Exception:
                pass
        return heuristic_entity_candidates(question, limit=8)

    def _structured_call(
        self,
        *,
        schema: type[ModelT],
        system_prompt: str,
        user_payload: str,
        model: str,
    ) -> ModelT:
        assert self.client is not None
        response = self.client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_payload},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "strict": True,
                    "schema": schema.model_json_schema(),
                },
            },
        )
        payload = response.choices[0].message.content or "{}"
        return schema.model_validate_json(payload)

    def _llm_extract_chunk(self, *, text: str, title: str, source_path: str) -> DocumentExtraction:
        prompt = (
            f"Document title: {title}\n"
            f"Source path: {source_path}\n\n"
            "Extract a graph-ready summary of the following text. Keep entities canonical, avoid duplicates, "
            "and only include explicit relationships supported by the text. Limit output to the most relevant facts.\n\n"
            f"Text:\n{text[:3500]}"
        )
        return self._structured_call(
            schema=DocumentExtraction,
            system_prompt=(
                "You are an enterprise knowledge graph ETL assistant. Return concise entities and relations that "
                "can be inserted into a Neo4j graph. Prefer stable canonical names and relation labels suitable "
                "for GraphRAG retrieval."
            ),
            user_payload=prompt,
            model=self.settings.groq_extraction_model,
        )

    def _heuristic_extract_chunk(self, text: str) -> DocumentExtraction:
        candidates = heuristic_entity_candidates(text, limit=10)
        entities = [
            ExtractedEntity(
                name=name,
                kind=self._infer_entity_kind(name),
                description=f"Heuristically extracted from chunk mentioning {name}.",
            )
            for name in candidates
        ]

        relations: list[ExtractedRelation] = []
        seen_relations: set[tuple[str, str, str]] = set()
        for sentence in sentence_split(text):
            lowered = sentence.lower()
            matched_label = next((label for phrase, label in RELATION_KEYWORDS.items() if phrase in lowered), None)
            if matched_label is None:
                continue
            sentence_entities = [entity for entity in candidates if entity in sentence]
            if len(sentence_entities) < 2:
                continue
            source = sentence_entities[0]
            target = sentence_entities[1]
            key = (normalize_key(source), normalize_key(target), matched_label)
            if key in seen_relations:
                continue
            seen_relations.add(key)
            relations.append(
                ExtractedRelation(
                    source=source,
                    target=target,
                    relation=matched_label,
                    evidence=truncate_text(sentence, 180),
                    confidence=0.5,
                )
            )
            if len(relations) >= 10:
                break

        summary = sentence_split(text)[0] if text.strip() else ""
        tags = [entity.kind for entity in entities[:3]]
        return DocumentExtraction(summary=summary, entities=entities, relationships=relations, salience_tags=tags)

    @staticmethod
    def _infer_entity_kind(name: str) -> str:
        lowered = name.lower()
        if "plant" in lowered:
            return "plant"
        if "team" in lowered:
            return "team"
        if "gateway" in lowered:
            return "gateway"
        if "camera" in lowered:
            return "camera"
        if "platform" in lowered:
            return "platform"
        if "lake" in lowered:
            return "data_store"
        if "standard" in lowered or "policy" in lowered:
            return "standard"
        if "incident" in lowered or "postmortem" in lowered:
            return "incident"
        if lowered.startswith("kb-") or "runbook" in lowered or "guide" in lowered:
            return "document"
        if "supplier" in lowered:
            return "supplier"
        return "concept"
