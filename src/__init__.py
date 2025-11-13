"""
Package initialization for RAG Chatbot.
"""

__version__ = "1.0.0"
__author__ = "RAG Chatbot Team"

from .config import *
from .document_ingestion import DocumentIngestor, ingest_document, ingest_documents
from .text_chunking import TextChunker, chunk_text, chunk_documents
from .vector_store import VectorStore, create_vector_store
from .rag_engine import RAGEngine, create_rag_engine
from .utils import *

__all__ = [
    "DocumentIngestor",
    "ingest_document",
    "ingest_documents",
    "TextChunker",
    "chunk_text",
    "chunk_documents",
    "VectorStore",
    "create_vector_store",
    "RAGEngine",
    "create_rag_engine",
]
