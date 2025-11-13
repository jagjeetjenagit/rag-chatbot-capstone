"""
Unit tests for RAG engine module.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile

from src.vector_store import VectorStore
from src.rag_engine import RAGEngine, create_rag_engine


@pytest.fixture
def temp_db_directory():
    """Create a temporary directory for test database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store."""
    mock_vs = Mock(spec=VectorStore)
    mock_vs.search.return_value = [
        {
            "text": "Machine learning is a subset of artificial intelligence.",
            "metadata": {"source": "test.txt", "chunk_index": 0},
            "similarity_score": 0.95
        },
        {
            "text": "Neural networks are inspired by biological brains.",
            "metadata": {"source": "test.txt", "chunk_index": 1},
            "similarity_score": 0.85
        }
    ]
    return mock_vs


@pytest.fixture
def mock_openai_response():
    """Create a mock OpenAI response."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Machine learning is a subset of AI that enables systems to learn from data."
    return mock_response


class TestRAGEngineInitialization:
    """Test RAG engine initialization."""
    
    def test_init_with_openai(self, mock_vector_store):
        """Test initialization with OpenAI."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'sk-test123'}):
            with patch('src.rag_engine.openai.OpenAI'):
                engine = RAGEngine(mock_vector_store, llm_provider="openai")
                assert engine.llm_provider == "openai"
                assert engine.vector_store == mock_vector_store
    
    def test_init_with_google(self, mock_vector_store):
        """Test initialization with Google AI."""
        with patch.dict('os.environ', {'GOOGLE_API_KEY': 'test-key'}):
            with patch('src.rag_engine.genai'):
                engine = RAGEngine(mock_vector_store, llm_provider="google")
                assert engine.llm_provider == "google"
    
    def test_init_invalid_provider(self, mock_vector_store):
        """Test initialization with invalid provider."""
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            RAGEngine(mock_vector_store, llm_provider="invalid")


class TestRAGEngineRetrieval:
    """Test retrieval functionality."""
    
    def test_retrieve_context(self, mock_vector_store):
        """Test context retrieval."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'sk-test123'}):
            with patch('src.rag_engine.openai.OpenAI'):
                engine = RAGEngine(mock_vector_store, llm_provider="openai", top_k=3)
                
                results = engine.retrieve_context("What is machine learning?")
                
                assert len(results) > 0
                mock_vector_store.search.assert_called_once()
    
    def test_retrieve_context_with_filter(self, mock_vector_store):
        """Test context retrieval with source filter."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'sk-test123'}):
            with patch('src.rag_engine.openai.OpenAI'):
                engine = RAGEngine(mock_vector_store, llm_provider="openai")
                
                results = engine.retrieve_context(
                    "test query",
                    filter_source="test.txt"
                )
                
                mock_vector_store.search.assert_called_once()
                call_args = mock_vector_store.search.call_args
                assert call_args[1]["filter_metadata"] == {"source": "test.txt"}
    
    def test_format_context(self, mock_vector_store):
        """Test context formatting."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'sk-test123'}):
            with patch('src.rag_engine.openai.OpenAI'):
                engine = RAGEngine(mock_vector_store, llm_provider="openai")
                
                chunks = [
                    {
                        "text": "Test text 1",
                        "metadata": {"source": "doc1.txt", "chunk_index": 0}
                    },
                    {
                        "text": "Test text 2",
                        "metadata": {"source": "doc2.txt", "chunk_index": 1}
                    }
                ]
                
                context = engine._format_context(chunks)
                
                assert "Test text 1" in context
                assert "Test text 2" in context
                assert "doc1.txt" in context
                assert "doc2.txt" in context
    
    def test_format_context_empty(self, mock_vector_store):
        """Test formatting empty context."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'sk-test123'}):
            with patch('src.rag_engine.openai.OpenAI'):
                engine = RAGEngine(mock_vector_store, llm_provider="openai")
                
                context = engine._format_context([])
                
                assert "No relevant context" in context


class TestRAGEngineAnswerGeneration:
    """Test answer generation."""
    
    def test_generate_answer_success(self, mock_vector_store, mock_openai_response):
        """Test successful answer generation."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'sk-test123'}):
            with patch('src.rag_engine.openai.OpenAI') as mock_openai:
                mock_client = MagicMock()
                mock_client.chat.completions.create.return_value = mock_openai_response
                mock_openai.return_value = mock_client
                
                engine = RAGEngine(mock_vector_store, llm_provider="openai")
                
                result = engine.generate_answer("What is machine learning?")
                
                assert result["answer"] is not None
                assert result["context_found"] is True
                assert "sources" in result
                assert len(result["sources"]) > 0
    
    def test_generate_answer_no_context(self, mock_vector_store):
        """Test answer generation when no context is found."""
        mock_vector_store.search.return_value = []
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'sk-test123'}):
            with patch('src.rag_engine.openai.OpenAI'):
                engine = RAGEngine(mock_vector_store, llm_provider="openai")
                
                result = engine.generate_answer("Unknown query")
                
                assert result["context_found"] is False
                assert "couldn't find" in result["answer"].lower()
                assert len(result["sources"]) == 0
    
    def test_chat_method(self, mock_vector_store, mock_openai_response):
        """Test chat method."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'sk-test123'}):
            with patch('src.rag_engine.openai.OpenAI') as mock_openai:
                mock_client = MagicMock()
                mock_client.chat.completions.create.return_value = mock_openai_response
                mock_openai.return_value = mock_client
                
                engine = RAGEngine(mock_vector_store, llm_provider="openai")
                
                result = engine.chat("What is AI?")
                
                assert "answer" in result
                assert result is not None


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_create_rag_engine(self, mock_vector_store):
        """Test create_rag_engine convenience function."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'sk-test123'}):
            with patch('src.rag_engine.openai.OpenAI'):
                engine = create_rag_engine(mock_vector_store, llm_provider="openai")
                
                assert engine is not None
                assert isinstance(engine, RAGEngine)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
