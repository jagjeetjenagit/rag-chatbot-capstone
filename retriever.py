"""
Retrieval Module for RAG Chatbot.
Performs semantic search on ChromaDB collection with fallback to keyword search.

Features:
- Semantic search using sentence-transformers embeddings
- Fallback to keyword search for low-similarity results
- Unit-testable functions
- Demo mode for testing
"""
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

# Import ChromaDB
try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    raise ImportError("chromadb not installed. Install with: pip install chromadb")

# Import shared client
from chroma_client import get_chroma_client

# Import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("sentence-transformers not installed. Install with: pip install sentence-transformers")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Retriever:
    """
    Handles semantic search and keyword search for document retrieval.
    """
    
    def __init__(
        self,
        chroma_db_path: str = "./chroma_db",
        collection_name: str = "capstone_docs",
        embedding_model: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.2
    ):
        """
        Initialize the retriever.
        
        Args:
            chroma_db_path: Path to ChromaDB persistent storage
            collection_name: Name of the Chroma collection
            embedding_model: SentenceTransformer model name
            similarity_threshold: Minimum average similarity for semantic search (default: 0.2)
        """
        self.chroma_db_path = Path(chroma_db_path)
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.similarity_threshold = similarity_threshold
        
        logger.info(f"Initializing Retriever: collection={collection_name}, threshold={similarity_threshold}")
        
        # Verify ChromaDB path exists
        if not self.chroma_db_path.exists():
            raise FileNotFoundError(
                f"ChromaDB path not found: {self.chroma_db_path}. "
                "Please run embeddings_and_chroma_setup.py first."
            )
        
        # Get shared ChromaDB client
        self.client = get_chroma_client(str(self.chroma_db_path))
        logger.info(f"Connected to ChromaDB at: {self.chroma_db_path}")
        
        # Get collection
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            doc_count = self.collection.count()
            logger.info(f"Loaded collection '{self.collection_name}' with {doc_count} documents")
        except Exception as e:
            raise ValueError(
                f"Collection '{self.collection_name}' not found. "
                "Please run embeddings_and_chroma_setup.py first."
            ) from e
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        logger.info(f"Embedding model loaded. Dimension: {self.embedding_model.get_sentence_embedding_dimension()}")
    
    def _embed_query(self, query: str) -> List[float]:
        """
        Embed a query string using the same embedding model.
        
        Args:
            query: Query string to embed
            
        Returns:
            List[float]: Embedding vector
        """
        embedding = self.embedding_model.encode(query, convert_to_numpy=True)
        return embedding.tolist()
    
    def _semantic_search(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """
        Perform semantic search using vector embeddings.
        
        Args:
            query: Query string
            k: Number of top results to return
            
        Returns:
            List[Dict]: List of results with id, text, metadata, and score
        """
        logger.info(f"Performing semantic search for: '{query}' (k={k})")
        
        # Embed the query
        query_embedding = self._embed_query(query)
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Format results
        formatted_results = []
        
        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                # Convert distance to similarity score (1 - normalized distance)
                # ChromaDB returns squared L2 distance, so we normalize it
                distance = results['distances'][0][i]
                # Similarity score: higher is better (inverse of distance)
                # For L2 distance, we convert to a 0-1 similarity score
                similarity = 1 / (1 + distance)
                
                result = {
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'score': round(similarity, 4)
                }
                formatted_results.append(result)
        
        # Calculate average similarity
        if formatted_results:
            avg_similarity = sum(r['score'] for r in formatted_results) / len(formatted_results)
            logger.info(f"Semantic search returned {len(formatted_results)} results, avg_similarity={avg_similarity:.4f}")
        else:
            avg_similarity = 0
            logger.warning("Semantic search returned no results")
        
        return formatted_results, avg_similarity
    
    def _keyword_search(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """
        Perform keyword search using substring matching.
        Fallback method when semantic search has low similarity.
        
        Args:
            query: Query string
            k: Number of top results to return
            
        Returns:
            List[Dict]: List of results with id, text, metadata, and score
        """
        logger.info(f"Performing keyword search for: '{query}' (k={k})")
        
        # Get all documents from collection
        all_docs = self.collection.get(
            include=['documents', 'metadatas']
        )
        
        if not all_docs['ids']:
            logger.warning("No documents in collection for keyword search")
            return []
        
        # Normalize query for case-insensitive matching
        query_lower = query.lower()
        query_terms = query_lower.split()
        
        # Score each document based on keyword matches
        scored_docs = []
        
        for i in range(len(all_docs['ids'])):
            doc_text = all_docs['documents'][i]
            doc_text_lower = doc_text.lower()
            
            # Count keyword matches
            score = 0
            
            # Check for full query substring match (higher score)
            if query_lower in doc_text_lower:
                score += 10
            
            # Check for individual term matches
            for term in query_terms:
                if len(term) > 2:  # Only count terms longer than 2 chars
                    score += doc_text_lower.count(term)
            
            if score > 0:
                scored_docs.append({
                    'id': all_docs['ids'][i],
                    'text': doc_text,
                    'metadata': all_docs['metadatas'][i] if all_docs['metadatas'] else {},
                    'score': score,
                    'raw_score': score  # Keep original score for sorting
                })
        
        # Sort by score descending
        scored_docs.sort(key=lambda x: x['raw_score'], reverse=True)
        
        # Take top k
        top_results = scored_docs[:k]
        
        # Normalize scores to 0-1 range for consistency
        if top_results:
            max_score = top_results[0]['raw_score']
            for result in top_results:
                result['score'] = round(result['raw_score'] / max_score, 4)
                del result['raw_score']
        
        logger.info(f"Keyword search returned {len(top_results)} results")
        return top_results
    
    def get_top_k(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most relevant chunks for a query.
        Uses semantic search with fallback to keyword search for low-similarity results.
        
        Args:
            query: Query string
            k: Number of top results to return (default: 4)
            
        Returns:
            List[Dict]: List of results with format:
                {"id": str, "text": str, "metadata": dict, "score": float}
        """
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return []
        
        try:
            # First, try semantic search
            semantic_results, avg_similarity = self._semantic_search(query, k)
            
            # Check if semantic search returned good results
            if avg_similarity >= self.similarity_threshold:
                logger.info(f"Using semantic search results (avg_similarity={avg_similarity:.4f} >= {self.similarity_threshold})")
                return semantic_results
            else:
                # Fallback to keyword search
                logger.info(
                    f"Semantic search similarity too low (avg={avg_similarity:.4f} < {self.similarity_threshold}), "
                    "falling back to keyword search"
                )
                keyword_results = self._keyword_search(query, k)
                
                if keyword_results:
                    logger.info(f"Using keyword search results ({len(keyword_results)} matches)")
                    return keyword_results
                else:
                    # Return semantic results even if similarity is low (better than nothing)
                    logger.info("No keyword matches found, returning semantic results despite low similarity")
                    return semantic_results
        
        except Exception as e:
            logger.error(f"Error during retrieval: {str(e)}", exc_info=True)
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dict[str, Any]: Collection statistics
        """
        return {
            'collection_name': self.collection_name,
            'total_documents': self.collection.count(),
            'chroma_db_path': str(self.chroma_db_path),
            'embedding_model': self.embedding_model_name,
            'similarity_threshold': self.similarity_threshold
        }


def get_top_k(
    query: str,
    k: int = 4,
    chroma_db_path: str = "./chroma_db",
    collection_name: str = "capstone_docs",
    similarity_threshold: float = 0.2
) -> List[Dict[str, Any]]:
    """
    Convenience function to retrieve top-k results without managing Retriever instance.
    
    Args:
        query: Query string
        k: Number of top results to return
        chroma_db_path: Path to ChromaDB storage
        collection_name: Name of the collection
        similarity_threshold: Minimum average similarity for semantic search
        
    Returns:
        List[Dict]: List of top-k results
    """
    retriever = Retriever(
        chroma_db_path=chroma_db_path,
        collection_name=collection_name,
        similarity_threshold=similarity_threshold
    )
    return retriever.get_top_k(query, k)


if __name__ == "__main__":
    """
    Demo: Test retriever with sample queries.
    """
    print("=" * 80)
    print("RETRIEVER DEMO")
    print("=" * 80)
    
    # Configuration
    CHROMA_DB_PATH = "./chroma_db"
    COLLECTION_NAME = "capstone_docs"
    
    # Check if ChromaDB exists
    if not Path(CHROMA_DB_PATH).exists():
        print(f"\n❌ Error: ChromaDB not found at {CHROMA_DB_PATH}")
        print("   Please run embeddings_and_chroma_setup.py first to create the database.")
        exit(1)
    
    try:
        # Initialize retriever
        print(f"\nInitializing retriever...")
        print(f"  ChromaDB Path: {CHROMA_DB_PATH}")
        print(f"  Collection: {COLLECTION_NAME}")
        print(f"  Similarity Threshold: 0.2")
        print("-" * 80)
        
        retriever = Retriever(
            chroma_db_path=CHROMA_DB_PATH,
            collection_name=COLLECTION_NAME,
            similarity_threshold=0.2
        )
        
        # Get collection stats
        stats = retriever.get_collection_stats()
        print(f"\nCollection Statistics:")
        print(f"  Total Documents: {stats['total_documents']}")
        print(f"  Embedding Model: {stats['embedding_model']}")
        print("-" * 80)
        
        # Test queries
        test_queries = [
            ("artificial intelligence machine learning", 4),  # Should find semantic matches
            ("RAG chatbot retrieval", 3),  # Should find semantic matches
            ("neural networks deep learning", 4),  # May need keyword fallback
            ("xyz123nonexistent", 2),  # Should test fallback behavior
        ]
        
        for idx, (query, k) in enumerate(test_queries, 1):
            print(f"\n{'=' * 80}")
            print(f"TEST QUERY {idx}: '{query}' (k={k})")
            print("=" * 80)
            
            results = retriever.get_top_k(query, k=k)
            
            if results:
                print(f"\nFound {len(results)} results:\n")
                
                for i, result in enumerate(results, 1):
                    print(f"Result {i}:")
                    print(f"  ID: {result['id']}")
                    print(f"  Score: {result['score']:.4f}")
                    print(f"  Source: {result['metadata'].get('source', 'unknown')}")
                    print(f"  Chunk Index: {result['metadata'].get('chunk_index', 'N/A')}")
                    print(f"  Text Preview: {result['text'][:150]}...")
                    print()
            else:
                print("\n⚠️  No results found")
            
            print("-" * 80)
        
        # Test different similarity thresholds
        print(f"\n{'=' * 80}")
        print("TESTING SIMILARITY THRESHOLD BEHAVIOR")
        print("=" * 80)
        
        test_query = "machine learning models"
        
        for threshold in [0.1, 0.3, 0.5]:
            print(f"\nQuery: '{test_query}' with threshold={threshold}")
            retriever_test = Retriever(
                chroma_db_path=CHROMA_DB_PATH,
                collection_name=COLLECTION_NAME,
                similarity_threshold=threshold
            )
            results = retriever_test.get_top_k(test_query, k=3)
            print(f"  Results: {len(results)} documents returned")
            if results:
                print(f"  Top score: {results[0]['score']:.4f}")
        
        print("\n" + "=" * 80)
        print("DEMO COMPLETE")
        print("=" * 80)
        print("\nRetriever is working correctly!")
        print("You can now use it in your RAG pipeline with:")
        print("  from retriever import get_top_k")
        print("  results = get_top_k('your query here', k=4)")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {str(e)}")
        logger.error("Demo failed", exc_info=True)
        exit(1)


# Export alias for backwards compatibility
DocumentRetriever = Retriever
