"""
Integration tests for the complete RAG pipeline.
"""
import pytest
import tempfile
from pathlib import Path

from src.document_ingestion import DocumentIngestor
from src.text_chunking import TextChunker
from src.vector_store import VectorStore


@pytest.fixture
def temp_directory():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_document(temp_directory):
    """Create a sample document for integration testing."""
    file_path = temp_directory / "sample.txt"
    content = """
    Artificial Intelligence is transforming the world. Machine learning is a subset of AI
    that enables computers to learn from data. Deep learning uses neural networks with
    multiple layers. Natural language processing helps computers understand human language.
    Computer vision allows machines to interpret visual information. Reinforcement learning
    teaches agents to make decisions through trial and error.
    """
    file_path.write_text(content, encoding='utf-8')
    return file_path


class TestEndToEndPipeline:
    """Integration tests for the complete RAG pipeline."""
    
    def test_full_pipeline(self, sample_document, temp_directory):
        """Test the complete pipeline: ingest -> chunk -> store -> search."""
        # Step 1: Ingest document
        ingestor = DocumentIngestor()
        text, metadata = ingestor.ingest_document(sample_document)
        
        assert text is not None
        assert len(text) > 0
        
        # Step 2: Chunk text
        chunker = TextChunker(
            chunk_size_min=100,
            chunk_size_max=200,
            overlap_percent=10
        )
        chunks = chunker.chunk_text(text, metadata["source"], metadata)
        
        assert len(chunks) > 0
        
        # Step 3: Store in vector database
        db_dir = temp_directory / "test_db"
        vector_store = VectorStore(
            collection_name="test_integration",
            persist_directory=str(db_dir)
        )
        
        added = vector_store.add_chunks(chunks)
        assert added == len(chunks)
        
        # Step 4: Search
        results = vector_store.search("machine learning", top_k=3)
        
        assert len(results) > 0
        assert any("machine learning" in r["text"].lower() for r in results)
    
    def test_multiple_documents_pipeline(self, temp_directory):
        """Test pipeline with multiple documents."""
        # Create multiple documents
        doc1 = temp_directory / "doc1.txt"
        doc2 = temp_directory / "doc2.txt"
        
        doc1.write_text("Python is a programming language. It's used for data science.")
        doc2.write_text("JavaScript is used for web development. It runs in browsers.")
        
        # Process both documents
        ingestor = DocumentIngestor()
        chunker = TextChunker(chunk_size_min=50, chunk_size_max=100)
        
        db_dir = temp_directory / "test_db"
        vector_store = VectorStore(
            collection_name="test_multi",
            persist_directory=str(db_dir)
        )
        
        all_chunks = []
        for doc in [doc1, doc2]:
            text, metadata = ingestor.ingest_document(doc)
            chunks = chunker.chunk_text(text, metadata["source"], metadata)
            all_chunks.extend(chunks)
        
        vector_store.add_chunks(all_chunks)
        
        # Verify we can search across documents
        results = vector_store.search("programming", top_k=5)
        assert len(results) > 0
        
        # Verify we can filter by source
        results_filtered = vector_store.search(
            "programming",
            filter_metadata={"source": "doc1.txt"}
        )
        if results_filtered:
            assert all(r["metadata"]["source"] == "doc1.txt" for r in results_filtered)
    
    def test_stats_and_management(self, sample_document, temp_directory):
        """Test database statistics and management functions."""
        # Set up pipeline
        ingestor = DocumentIngestor()
        chunker = TextChunker()
        
        db_dir = temp_directory / "test_db"
        vector_store = VectorStore(
            collection_name="test_management",
            persist_directory=str(db_dir)
        )
        
        # Add document
        text, metadata = ingestor.ingest_document(sample_document)
        chunks = chunker.chunk_text(text, metadata["source"], metadata)
        vector_store.add_chunks(chunks)
        
        # Test stats
        stats = vector_store.get_stats()
        assert stats["total_chunks"] > 0
        assert stats["unique_sources"] == 1
        assert metadata["source"] in stats["sources"]
        
        # Test delete by source
        deleted = vector_store.delete_by_source(metadata["source"])
        assert deleted > 0
        assert vector_store.collection.count() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
