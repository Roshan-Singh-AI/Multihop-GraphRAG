from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings
from .loaders import LoadedDocument
from .utils import chunk_ref



def chunk_documents(documents: list[LoadedDocument], settings: Settings) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n## ", "\n\n", "\n", ". ", " "],
    )

    chunks: list[Document] = []
    for document in documents:
        pieces = splitter.split_text(document.text)
        for index, piece in enumerate(pieces):
            chunks.append(
                Document(
                    page_content=piece,
                    metadata={
                        "doc_id": document.doc_id,
                        "title": document.title,
                        "source_path": document.source_path,
                        "chunk_index": index,
                        "chunk_id": chunk_ref(document.doc_id, index),
                    },
                )
            )
    return chunks
