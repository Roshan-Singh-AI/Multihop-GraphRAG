from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings
from .loaders import LoadedDocument
from .utils import chunk_ref



def chunk_documents(documents: list[LoadedDocument], settings: Settings) -> list[Document]:    """Split documents into smaller chunks for embedding and retrieval.
    
    Args:
        documents: List of loaded documents to chunk.
        settings: Application settings containing chunk size and overlap.
        
    Returns:
        List of Document objects with chunk metadata.
    """
    """Split documents into smaller chunks for embedding and retrieval.
    
    Args:
        documents: List of loaded documents to chunk.
        settings: Application settings containing chunk size and overlap.
        
    Returns:
        List of Document objects with chunk metadata.
    """

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



def validate_chunk_size(chunk_size: int, overlap: int) -> bool:
    """Validate chunk size and overlap parameters.
    
    Args:
        chunk_size: Target size for text chunks.
        overlap: Number of characters to overlap between chunks.
        
    Returns:
        True if parameters are valid, False otherwise.
    """
    if chunk_size <= 0:
        return False
    if overlap < 0:
        return False
    if overlap >= chunk_size:
        return False
    return True



def validate_chunk_size(chunk_size: int, overlap: int) -> bool:
    """Validate chunk size and overlap parameters.
    
    Args:
        chunk_size: Target size for text chunks.
        overlap: Number of characters to overlap between chunks.
        
    Returns:
        True if parameters are valid, False otherwise.
    """
    if chunk_size <= 0:
        return False
    if overlap < 0:
        return False
    if overlap >= chunk_size:
        return False
    return True
