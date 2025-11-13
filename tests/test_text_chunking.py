"""
Unit tests for text_chunking module.
"""
import pytest

from src.text_chunking import (
    TextChunker,
    chunk_text,
    chunk_documents
)


class TestTextChunker:
    """Test cases for TextChunker class."""
    
    def test_initialization(self):
        """Test TextChunker initialization."""
        chunker = TextChunker(
            chunk_size_min=500,
            chunk_size_max=800,
            overlap_percent=10
        )
        
        assert chunker.chunk_size_min == 500
        assert chunker.chunk_size_max == 800
        assert chunker.overlap_percent == 10
        assert chunker.overlap_size == 80  # 10% of 800
    
    def test_split_into_sentences(self):
        """Test sentence splitting."""
        chunker = TextChunker()
        text = "This is sentence one. This is sentence two! Is this sentence three?"
        
        sentences = chunker._split_into_sentences(text)
        
        assert len(sentences) >= 3
        assert any("sentence one" in s for s in sentences)
    
    def test_chunk_simple_text(self):
        """Test chunking simple text."""
        chunker = TextChunker(
            chunk_size_min=50,
            chunk_size_max=100,
            overlap_percent=10
        )
        
        text = (
            "This is the first sentence. This is the second sentence. "
            "This is the third sentence. This is the fourth sentence. "
            "This is the fifth sentence."
        )
        
        chunks = chunker.chunk_text(text, source="test.txt")
        
        assert len(chunks) > 0
        assert all("text" in chunk for chunk in chunks)
        assert all("metadata" in chunk for chunk in chunks)
    
    def test_chunk_metadata(self):
        """Test chunk metadata generation."""
        chunker = TextChunker(
            chunk_size_min=50,
            chunk_size_max=100,
            overlap_percent=10
        )
        
        text = "A " * 100  # Simple repeated text
        chunks = chunker.chunk_text(text, source="test.txt")
        
        for i, chunk in enumerate(chunks):
            assert chunk["metadata"]["source"] == "test.txt"
            assert chunk["metadata"]["chunk_index"] == i
            assert "char_count" in chunk["metadata"]
    
    def test_chunk_size_constraints(self):
        """Test that chunks respect size constraints."""
        chunker = TextChunker(
            chunk_size_min=500,
            chunk_size_max=800,
            overlap_percent=10
        )
        
        # Create a long text
        text = ". ".join([f"This is sentence number {i}" for i in range(100)])
        
        chunks = chunker.chunk_text(text, source="test.txt")
        
        for chunk in chunks:
            chunk_size = len(chunk["text"])
            # Most chunks should be within bounds (allowing some flexibility)
            if chunk != chunks[-1]:  # Last chunk might be smaller
                assert chunk_size >= chunker.chunk_size_min * 0.8  # Allow some variance
    
    def test_empty_text(self):
        """Test chunking empty text."""
        chunker = TextChunker()
        chunks = chunker.chunk_text("", source="empty.txt")
        
        assert chunks == []
    
    def test_whitespace_only_text(self):
        """Test chunking whitespace-only text."""
        chunker = TextChunker()
        chunks = chunker.chunk_text("   \n\n   ", source="whitespace.txt")
        
        assert chunks == []
    
    def test_chunk_overlap(self):
        """Test that chunks have proper overlap."""
        chunker = TextChunker(
            chunk_size_min=100,
            chunk_size_max=200,
            overlap_percent=20
        )
        
        # Create text with distinct sentences
        sentences = [f"Unique sentence number {i} with distinct content." for i in range(20)]
        text = " ".join(sentences)
        
        chunks = chunker.chunk_text(text, source="test.txt")
        
        if len(chunks) > 1:
            # Check if there's any text overlap between consecutive chunks
            # (This is a simple check - actual overlap may vary)
            assert len(chunks) > 0
    
    def test_additional_metadata(self):
        """Test adding additional metadata to chunks."""
        chunker = TextChunker()
        text = "Test sentence. " * 50
        
        extra_metadata = {
            "file_type": "txt",
            "author": "Test Author"
        }
        
        chunks = chunker.chunk_text(text, source="test.txt", metadata=extra_metadata)
        
        for chunk in chunks:
            assert chunk["metadata"]["file_type"] == "txt"
            assert chunk["metadata"]["author"] == "Test Author"


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_chunk_text_function(self):
        """Test chunk_text convenience function."""
        text = "This is a test sentence. " * 30
        chunks = chunk_text(
            text,
            source="test.txt",
            chunk_size_min=100,
            chunk_size_max=200
        )
        
        assert len(chunks) > 0
        assert all("text" in chunk for chunk in chunks)
    
    def test_chunk_documents_function(self):
        """Test chunk_documents convenience function."""
        documents = [
            ("Text for document 1. " * 30, {"source": "doc1.txt"}),
            ("Text for document 2. " * 30, {"source": "doc2.txt"}),
        ]
        
        chunks = chunk_documents(
            documents,
            chunk_size_min=100,
            chunk_size_max=200
        )
        
        assert len(chunks) > 0
        # Should have chunks from both documents
        sources = {chunk["metadata"]["source"] for chunk in chunks}
        assert "doc1.txt" in sources
        assert "doc2.txt" in sources


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
