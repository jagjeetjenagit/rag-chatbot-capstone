"""
Vector Store Module using ChromaDB.
Manages embeddings, storage, and retrieval of document chunks.
"""
import os
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from .config import (
    EMBEDDING_MODEL,
    CHROMA_DB_DIR,
    CHROMA_COLLECTION_NAME,
    TOP_K_RETRIEVAL
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    """
    Manages vector embeddings and similarity search using ChromaDB.
    Uses sentence-transformers for generating embeddings.
    """
    
    def __init__(
        self,
        collection_name: str = CHROMA_COLLECTION_NAME,
        persist_directory: str = None,
        embedding_model_name: str = EMBEDDING_MODEL
    ):
        """
        Initialize the vector store.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database
            embedding_model_name: Name of the sentence-transformers model
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory or str(CHROMA_DB_DIR)
        self.embedding_model_name = embedding_model_name
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        logger.info(f"Embedding model loaded. Dimension: {self.embedding_model.get_sentence_embedding_dimension()}")
        
        # Initialize ChromaDB client
        self._initialize_client()
        
    def _initialize_client(self):
        """Initialize ChromaDB client and collection."""
        try:
            # Create persist directory if it doesn't exist
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB client with persistence
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "RAG document chunks with embeddings"}
            )
            
            logger.info(
                f"ChromaDB initialized. Collection: {self.collection_name}, "
                f"Documents: {self.collection.count()}"
            )
            
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {str(e)}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            embeddings = self.embedding_model.encode(
                texts,
                show_progress_bar=len(texts) > 10,
                convert_to_numpy=True
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Add document chunks to the vector store.
        
        Args:
            chunks: List of chunk dictionaries with 'text' and 'metadata' keys
            
        Returns:
            int: Number of chunks added
        """
        if not chunks:
            logger.warning("No chunks provided to add to vector store")
            return 0
        
        try:
            # Extract texts and metadata
            texts = [chunk["text"] for chunk in chunks]
            metadatas = [chunk["metadata"] for chunk in chunks]
            
            # Generate unique IDs for each chunk
            # Format: source_chunkindex_timestamp
            import time
            timestamp = int(time.time())
            ids = [
                f"{meta['source']}_{meta['chunk_index']}_{timestamp}"
                for meta in metadatas
            ]
            
            # Convert all metadata values to strings (ChromaDB requirement)
            metadatas = [
                {k: str(v) for k, v in meta.items()}
                for meta in metadatas
            ]
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(texts)} chunks...")
            embeddings = self.generate_embeddings(texts)
            
            # Add to ChromaDB collection
            logger.info(f"Adding {len(texts)} chunks to ChromaDB...")
            self.collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(
                f"Successfully added {len(chunks)} chunks to vector store. "
                f"Total documents: {self.collection.count()}"
            )
            
            return len(chunks)
            
        except Exception as e:
            logger.error(f"Error adding chunks to vector store: {str(e)}")
            raise
    
    def search(
        self,
        query: str,
        top_k: int = TOP_K_RETRIEVAL,
        filter_metadata: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using semantic similarity.
        
        Args:
            query: Query string to search for
            top_k: Number of top results to return (default: 5)
            filter_metadata: Optional metadata filter (e.g., {"source": "doc.pdf"})
            
        Returns:
            List of dictionaries containing matching chunks with scores
        """
        try:
            # Check if collection has documents
            if self.collection.count() == 0:
                logger.warning("Vector store is empty. No documents to search.")
                return []
            
            # Generate query embedding
            query_embedding = self.generate_embeddings([query])[0]
            
            # Perform search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self.collection.count()),
                where=filter_metadata,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            
            if results and results["documents"] and results["documents"][0]:
                for i in range(len(results["documents"][0])):
                    result = {
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i],
                        # Convert distance to similarity score (0-1, higher is better)
                        "similarity_score": 1 / (1 + results["distances"][0][i])
                    }
                    formatted_results.append(result)
                
                logger.info(f"Search returned {len(formatted_results)} results for query: '{query[:50]}...'")
            else:
                logger.info(f"No results found for query: '{query[:50]}...'")
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            raise
    
    def get_all_sources(self) -> List[str]:
        """
        Get list of all unique source documents in the vector store.
        
        Returns:
            List of source document names
        """
        try:
            # Get all documents with metadata
            results = self.collection.get(include=["metadatas"])
            
            if not results or not results["metadatas"]:
                return []
            
            # Extract unique sources
            sources = set()
            for metadata in results["metadatas"]:
                if "source" in metadata:
                    sources.add(metadata["source"])
            
            return sorted(list(sources))
            
        except Exception as e:
            logger.error(f"Error getting sources: {str(e)}")
            return []
    
    def delete_by_source(self, source: str) -> int:
        """
        Delete all chunks from a specific source document.
        
        Args:
            source: Source document name to delete
            
        Returns:
            int: Number of chunks deleted
        """
        try:
            # Get all IDs for this source
            results = self.collection.get(
                where={"source": source},
                include=["metadatas"]
            )
            
            if results and results["ids"]:
                count = len(results["ids"])
                self.collection.delete(ids=results["ids"])
                logger.info(f"Deleted {count} chunks from source: {source}")
                return count
            else:
                logger.info(f"No chunks found for source: {source}")
                return 0
                
        except Exception as e:
            logger.error(f"Error deleting chunks: {str(e)}")
            raise
    
    def clear(self):
        """Clear all documents from the vector store."""
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "RAG document chunks with embeddings"}
            )
            logger.info("Vector store cleared")
        except Exception as e:
            logger.error(f"Error clearing vector store: {str(e)}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with statistics
        """
        try:
            total_chunks = self.collection.count()
            sources = self.get_all_sources()
            
            stats = {
                "total_chunks": total_chunks,
                "unique_sources": len(sources),
                "sources": sources,
                "collection_name": self.collection_name,
                "embedding_model": self.embedding_model_name,
                "persist_directory": self.persist_directory
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {}


# Convenience functions
def create_vector_store(
    collection_name: str = CHROMA_COLLECTION_NAME,
    persist_directory: str = None
) -> VectorStore:
    """
    Create and return a VectorStore instance.
    
    Args:
        collection_name: Name of the collection
        persist_directory: Directory to persist data
        
    Returns:
        VectorStore instance
    """
    return VectorStore(collection_name, persist_directory)
