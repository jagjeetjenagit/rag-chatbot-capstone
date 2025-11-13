"""
Unit tests for retriever.py module.
Tests semantic search, keyword fallback, and edge cases.
"""
import pytest
from pathlib import Path
from retriever import Retriever, get_top_k


class TestRetriever:
    """Test cases for Retriever class."""
    
    @pytest.fixture
    def retriever(self):
        """Create a Retriever instance for testing."""
        return Retriever(
            chroma_db_path="./chroma_db",
            collection_name="capstone_docs",
            similarity_threshold=0.2
        )
    
    def test_retriever_initialization(self, retriever):
        """Test that retriever initializes correctly."""
        assert retriever is not None
        assert retriever.collection_name == "capstone_docs"
        assert retriever.similarity_threshold == 0.2
        assert retriever.embedding_model is not None
    
    def test_get_top_k_returns_list(self, retriever):
        """Test that get_top_k returns a list."""
        results = retriever.get_top_k("test query", k=2)
        assert isinstance(results, list)
    
    def test_get_top_k_result_format(self, retriever):
        """Test that results have correct format."""
        results = retriever.get_top_k("RAG chatbot", k=2)
        
        if results:  # If we have results
            result = results[0]
            assert 'id' in result
            assert 'text' in result
            assert 'metadata' in result
            assert 'score' in result
            assert isinstance(result['score'], (int, float))
            assert 0 <= result['score'] <= 1
    
    def test_get_top_k_respects_k_parameter(self, retriever):
        """Test that k parameter limits results correctly."""
        for k in [1, 2, 3]:
            results = retriever.get_top_k("test", k=k)
            assert len(results) <= k
    
    def test_empty_query(self, retriever):
        """Test that empty query returns empty list."""
        results = retriever.get_top_k("", k=4)
        assert results == []
        
        results = retriever.get_top_k("   ", k=4)
        assert results == []
    
    def test_semantic_search(self, retriever):
        """Test semantic search functionality."""
        # Query that should trigger semantic search
        results = retriever.get_top_k("document processing pipeline", k=2)
        assert isinstance(results, list)
        # Should have results from semantic search
        assert len(results) > 0
    
    def test_keyword_fallback(self):
        """Test keyword fallback with high threshold."""
        # Create retriever with high threshold to trigger fallback
        retriever = Retriever(
            chroma_db_path="./chroma_db",
            collection_name="capstone_docs",
            similarity_threshold=0.95  # Very high threshold
        )
        
        # Query with specific keyword that exists in documents
        results = retriever.get_top_k("chunk", k=2)
        assert isinstance(results, list)
    
    def test_get_collection_stats(self, retriever):
        """Test collection statistics retrieval."""
        stats = retriever.get_collection_stats()
        
        assert 'collection_name' in stats
        assert 'total_documents' in stats
        assert 'chroma_db_path' in stats
        assert 'embedding_model' in stats
        assert 'similarity_threshold' in stats
        
        assert stats['collection_name'] == "capstone_docs"
        assert stats['total_documents'] >= 0
        assert stats['embedding_model'] == "all-MiniLM-L6-v2"


class TestConvenienceFunction:
    """Test the get_top_k convenience function."""
    
    def test_get_top_k_convenience_function(self):
        """Test that convenience function works."""
        results = get_top_k("test query", k=2)
        assert isinstance(results, list)
    
    def test_convenience_function_with_custom_params(self):
        """Test convenience function with custom parameters."""
        results = get_top_k(
            "test",
            k=3,
            chroma_db_path="./chroma_db",
            collection_name="capstone_docs",
            similarity_threshold=0.3
        )
        assert isinstance(results, list)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_invalid_chroma_path(self):
        """Test that invalid ChromaDB path raises error."""
        with pytest.raises(FileNotFoundError):
            Retriever(chroma_db_path="./nonexistent_path")
    
    def test_invalid_collection_name(self):
        """Test that invalid collection name raises error."""
        with pytest.raises(ValueError):
            Retriever(
                chroma_db_path="./chroma_db",
                collection_name="nonexistent_collection"
            )
    
    def test_large_k_value(self):
        """Test with k larger than available documents."""
        retriever = Retriever()
        results = retriever.get_top_k("test", k=1000)
        assert isinstance(results, list)
        # Should return all available documents, not crash
        assert len(results) <= 1000


if __name__ == "__main__":
    """Run tests with pytest."""
    print("=" * 80)
    print("RUNNING RETRIEVER UNIT TESTS")
    print("=" * 80)
    print("\nNote: Make sure ChromaDB is set up before running tests.")
    print("Run: python embeddings_and_chroma_setup.py\n")
    print("-" * 80)
    
    # Run pytest
    pytest.main([__file__, "-v", "--tb=short"])
