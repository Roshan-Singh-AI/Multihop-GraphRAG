from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from .utils import display_path, normalize_key

SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf", ".json"}


@dataclass(slots=True)
class LoadedDocument:
    doc_id: str
    title: str
    text: str
    source_path: str


class UnsupportedDocumentError(RuntimeError):
    pass



def discover_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)



def markdown_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        if line.strip().startswith("#"):
            return line.lstrip("# ").strip()
    return fallback



def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)



def read_json_document(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if "content" in payload and isinstance(payload["content"], str):
            return payload["content"]
        if "text" in payload and isinstance(payload["text"], str):
            return payload["text"]
    return json.dumps(payload, indent=2, ensure_ascii=False)



def load_document(path: Path, root: Path) -> LoadedDocument:
    suffix = path.suffix.lower()
    relative = display_path(path.relative_to(root if root.is_dir() else path.parent))
    fallback_title = path.stem.replace("_", " ").replace("-", " ").title()

    if suffix in {".md", ".txt"}:
        text = path.read_text(encoding="utf-8")
    elif suffix == ".pdf":
        text = read_pdf(path)
    elif suffix == ".json":
        text = read_json_document(path)
    else:
        raise UnsupportedDocumentError(f"Unsupported extension: {suffix}")

    title = markdown_title(text, fallback_title)
    doc_id = normalize_key(relative.rsplit(".", 1)[0])
    return LoadedDocument(doc_id=doc_id, title=title, text=text.strip(), source_path=relative)



def load_documents(path: Path) -> list[LoadedDocument]:
    files = discover_files(path)
    if not files:
        raise FileNotFoundError(f"No supported files found under {path}")

    root = path if path.is_dir() else path.parent
    documents = [load_document(file_path, root) for file_path in files]
    return [document for document in documents if document.text]
