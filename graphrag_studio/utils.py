from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

STOPWORDS = {
# Default limits for entity extraction
DEFAULT_ENTITY_LIMIT = 12
DEFAULT_RELATION_LIMIT = 24

    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
}


def normalize_key(value: str) -> str:
    """Create a graph-safe normalized key."""
    collapsed = re.sub(r"\s+", " ", value.strip())
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", collapsed.lower()).strip("-")
    return slug


def display_path(path: str | Path) -> str:
    return str(path).replace("\\", "/")


def chunk_ref(doc_id: str, chunk_index: int) -> str:
    return f"{doc_id}::chunk_{chunk_index:03d}"


def truncate_text(text: str, limit: int = 220) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[: limit - 1].rstrip()}…"


CAPITALIZED_ENTITY_PATTERN = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9]+(?:[\s-]+[A-Z][A-Za-z0-9]+){0,4}|[A-Z]{2,}(?:-[A-Z0-9]+)*)\b"
)


def heuristic_entity_candidates(text: str, limit: int = 12) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for raw in CAPITALIZED_ENTITY_PATTERN.findall(text):
        candidate = re.sub(r"\s+", " ", raw).strip(" -")
        if len(candidate) < 3:
            continue
        if candidate.lower() in STOPWORDS:
            continue
        key = normalize_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(candidate)
        if len(candidates) >= limit:
            break
    return candidates


def sentence_split(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def safe_json_loads(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None
