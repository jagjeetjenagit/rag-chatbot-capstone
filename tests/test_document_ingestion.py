"""
Unit tests for document_ingestion module.
"""
import pytest
from pathlib import Path
import tempfile
import os

from src.document_ingestion import (
    DocumentIngestor,
    DocumentIngestionError,
    ingest_document
)


@pytest.fixture
def temp_directory():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_txt_file(temp_directory):
    """Create a sample TXT file."""
    file_path = temp_directory / "test.txt"
    content = "This is a test document.\nIt contains multiple lines.\nFor testing purposes."
    file_path.write_text(content, encoding='utf-8')
    return file_path


@pytest.fixture
def sample_empty_txt_file(temp_directory):
    """Create an empty TXT file."""
    file_path = temp_directory / "empty.txt"
    file_path.write_text("", encoding='utf-8')
    return file_path


@pytest.fixture
def large_txt_file(temp_directory):
    """Create a large TXT file."""
    file_path = temp_directory / "large.txt"
    # Create a file larger than the max size
    content = "x" * (60 * 1024 * 1024)  # 60MB
    file_path.write_text(content, encoding='utf-8')
    return file_path


class TestDocumentIngestor:
    """Test cases for DocumentIngestor class."""
    
    def test_initialization(self):
        """Test DocumentIngestor initialization."""
        ingestor = DocumentIngestor()
        assert ingestor is not None
        assert len(ingestor.supported_extensions) > 0
        assert ingestor.max_file_size_mb > 0
    
    def test_validate_file_exists(self, sample_txt_file):
        """Test file validation for existing file."""
        ingestor = DocumentIngestor()
        assert ingestor.validate_file(sample_txt_file) is True
    
    def test_validate_file_not_exists(self):
        """Test file validation for non-existent file."""
        ingestor = DocumentIngestor()
        with pytest.raises(DocumentIngestionError, match="File not found"):
            ingestor.validate_file("nonexistent.txt")
    
    def test_validate_unsupported_extension(self, temp_directory):
        """Test file validation for unsupported extension."""
        file_path = temp_directory / "test.xyz"
        file_path.write_text("test")
        
        ingestor = DocumentIngestor()
        with pytest.raises(DocumentIngestionError, match="Unsupported file type"):
            ingestor.validate_file(file_path)
    
    def test_validate_file_too_large(self, large_txt_file):
        """Test file validation for files exceeding size limit."""
        ingestor = DocumentIngestor()
        with pytest.raises(DocumentIngestionError, match="File too large"):
            ingestor.validate_file(large_txt_file)
    
    def test_ingest_txt_success(self, sample_txt_file):
        """Test successful TXT file ingestion."""
        ingestor = DocumentIngestor()
        text = ingestor.ingest_txt(sample_txt_file)
        
        assert text is not None
        assert len(text) > 0
        assert "test document" in text
    
    def test_ingest_txt_empty_file(self, sample_empty_txt_file):
        """Test TXT ingestion with empty file."""
        ingestor = DocumentIngestor()
        with pytest.raises(DocumentIngestionError, match="empty"):
            ingestor.ingest_txt(sample_empty_txt_file)
    
    def test_ingest_document_txt(self, sample_txt_file):
        """Test full document ingestion for TXT."""
        ingestor = DocumentIngestor()
        text, metadata = ingestor.ingest_document(sample_txt_file)
        
        assert text is not None
        assert len(text) > 0
        assert metadata is not None
        assert metadata["source"] == sample_txt_file.name
        assert metadata["file_type"] == "txt"
        assert metadata["char_count"] == len(text)
    
    def test_convenience_function(self, sample_txt_file):
        """Test convenience function for document ingestion."""
        text, metadata = ingest_document(sample_txt_file)
        
        assert text is not None
        assert metadata is not None
        assert metadata["source"] == sample_txt_file.name


class TestMultipleDocuments:
    """Test cases for handling multiple documents."""
    
    def test_ingest_multiple_documents(self, temp_directory):
        """Test ingesting multiple documents."""
        from src.document_ingestion import ingest_documents
        
        # Create multiple test files
        file1 = temp_directory / "doc1.txt"
        file2 = temp_directory / "doc2.txt"
        file1.write_text("Document 1 content")
        file2.write_text("Document 2 content")
        
        results = ingest_documents([file1, file2])
        
        assert len(results) == 2
        assert all(isinstance(r, tuple) for r in results)
        assert all(len(r) == 2 for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
