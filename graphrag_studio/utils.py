from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

STOPWORDS = {
# Default limits for entity extraction
DEFAULT_ENTITY_LIMIT = 12

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
    # Handle edge cases for empty or whitespace-only input
    if not value or not value.strip():
        return ""

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


def format_iso_timestamp(dt: datetime | None = None) -> str:
    """Format datetime as ISO 8601 string for logging and debugging.
    
    Args:
        dt: Datetime object to format. Uses current UTC time if None.
        
    Returns:
        ISO 8601 formatted timestamp string.
    """
    from datetime import datetime, timezone
    if dt is None:
        dt = datetime.now(timezone.utc)
    return dt.isoformat(timespec="seconds")



def is_valid_entity_name(name: str, min_length: int = 2) -> bool:
    """Check if an entity name is valid for graph storage.
    
    Args:
        name: Entity name to validate.
        min_length: Minimum required length (default: 2).
        
    Returns:
        True if the entity name is valid.
    """
    if not name or len(name.strip()) < min_length:
        return False
    normalized = normalize_key(name)
    return len(normalized) >= min_length



def normalize_file_path(path: str | Path) -> str:
    """Normalize file path for consistent cross-platform handling.
    
    Args:
        path: File path to normalize.
        
    Returns:
        Normalized path string using forward slashes.
    """
    return str(Path(path).resolve()).replace("\\", "/")



def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default on zero division.
    
    Args:
        numerator: The dividend.
        denominator: The divisor.
        default: Value to return if denominator is zero.
        
    Returns:
        Result of division or default value.
    """
    if denominator == 0:
        return default
    return numerator / denominator



def format_result_count(count: int, singular: str, plural: str | None = None) -> str:
    """Format a result count with proper singular/plural form.
    
    Args:
        count: The number of items.
        singular: Singular form of the noun.
        plural: Plural form (default: singular + 's').
        
    Returns:
        Formatted string like "1 entity" or "5 entities".
    """
    if plural is None:
        plural = f"{singular}s"
    return f"{count} {singular if count == 1 else plural}"



def debug_chunk_info(chunk_id: str, text: str, max_preview: int = 80) -> str:
    """Generate a debug string for chunk inspection.
    
    Args:
        chunk_id: The chunk identifier.
        text: The chunk text content.
        max_preview: Maximum characters to include in preview.
        
    Returns:
        Debug string with chunk ID, length, and text preview.
    """
    preview = text[:max_preview].replace("\n", " ")
    if len(text) > max_preview:
        preview += "..."
    return f"[{chunk_id}] ({len(text)} chars): {preview}"
