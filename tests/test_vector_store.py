"""
Unit tests for vector_store module.
"""
import pytest
import tempfile
from pathlib import Path

from src.vector_store import VectorStore, create_vector_store


@pytest.fixture
def temp_db_directory():
    """Create a temporary directory for test database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def vector_store(temp_db_directory):
    """Create a VectorStore instance for testing."""
    return VectorStore(
        collection_name="test_collection",
        persist_directory=temp_db_directory
    )


@pytest.fixture
def sample_chunks():
    """Create sample chunks for testing."""
    return [
        {
            "text": "This is the first test chunk about machine learning.",
            "metadata": {
                "source": "test1.txt",
                "chunk_index": 0,
                "char_count": 52
            }
        },
        {
            "text": "This is the second test chunk about artificial intelligence.",
            "metadata": {
                "source": "test1.txt",
                "chunk_index": 1,
                "char_count": 61
            }
        },
        {
            "text": "This is a third chunk discussing natural language processing.",
            "metadata": {
                "source": "test2.txt",
                "chunk_index": 0,
                "char_count": 62
            }
        }
    ]


class TestVectorStore:
    """Test cases for VectorStore class."""
    
    def test_initialization(self, temp_db_directory):
        """Test VectorStore initialization."""
        vs = VectorStore(
            collection_name="test_collection",
            persist_directory=temp_db_directory
        )
        
        assert vs is not None
        assert vs.collection is not None
        assert vs.embedding_model is not None
    
    def test_generate_embeddings(self, vector_store):
        """Test embedding generation."""
        texts = ["This is a test.", "Another test text."]
        embeddings = vector_store.generate_embeddings(texts)
        
        assert len(embeddings) == 2
        assert all(isinstance(emb, list) for emb in embeddings)
        assert all(len(emb) > 0 for emb in embeddings)
    
    def test_add_chunks(self, vector_store, sample_chunks):
        """Test adding chunks to vector store."""
        count = vector_store.add_chunks(sample_chunks)
        
        assert count == len(sample_chunks)
        assert vector_store.collection.count() == len(sample_chunks)
    
    def test_add_empty_chunks(self, vector_store):
        """Test adding empty chunks list."""
        count = vector_store.add_chunks([])
        
        assert count == 0
    
    def test_search_basic(self, vector_store, sample_chunks):
        """Test basic search functionality."""
        # Add chunks first
        vector_store.add_chunks(sample_chunks)
        
        # Search
        results = vector_store.search("machine learning", top_k=2)
        
        assert len(results) <= 2
        assert all("text" in r for r in results)
        assert all("metadata" in r for r in results)
        assert all("similarity_score" in r for r in results)
    
    def test_search_empty_store(self, vector_store):
        """Test search on empty vector store."""
        results = vector_store.search("test query")
        
        assert results == []
    
    def test_search_with_filter(self, vector_store, sample_chunks):
        """Test search with metadata filter."""
        vector_store.add_chunks(sample_chunks)
        
        results = vector_store.search(
            "test",
            top_k=5,
            filter_metadata={"source": "test1.txt"}
        )
        
        # Should only return chunks from test1.txt
        if results:
            assert all(r["metadata"]["source"] == "test1.txt" for r in results)
    
    def test_get_all_sources(self, vector_store, sample_chunks):
        """Test getting all source documents."""
        vector_store.add_chunks(sample_chunks)
        
        sources = vector_store.get_all_sources()
        
        assert len(sources) > 0
        assert "test1.txt" in sources
        assert "test2.txt" in sources
    
    def test_delete_by_source(self, vector_store, sample_chunks):
        """Test deleting chunks by source."""
        vector_store.add_chunks(sample_chunks)
        initial_count = vector_store.collection.count()
        
        deleted = vector_store.delete_by_source("test1.txt")
        
        assert deleted > 0
        assert vector_store.collection.count() < initial_count
    
    def test_clear(self, vector_store, sample_chunks):
        """Test clearing the vector store."""
        vector_store.add_chunks(sample_chunks)
        assert vector_store.collection.count() > 0
        
        vector_store.clear()
        
        assert vector_store.collection.count() == 0
    
    def test_get_stats(self, vector_store, sample_chunks):
        """Test getting statistics."""
        vector_store.add_chunks(sample_chunks)
        
        stats = vector_store.get_stats()
        
        assert "total_chunks" in stats
        assert "unique_sources" in stats
        assert "sources" in stats
        assert stats["total_chunks"] == len(sample_chunks)
        assert stats["unique_sources"] == 2  # test1.txt and test2.txt


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_create_vector_store(self, temp_db_directory):
        """Test create_vector_store convenience function."""
        vs = create_vector_store(
            collection_name="test",
            persist_directory=temp_db_directory
        )
        
        assert vs is not None
        assert isinstance(vs, VectorStore)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
