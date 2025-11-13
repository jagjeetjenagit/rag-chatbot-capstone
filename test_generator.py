"""
Unit tests for generator.py module.
Tests answer generation, LLM integration, and fallback mechanisms.
"""
import pytest
from generator import (
    generate_answer,
    rule_based_generate,
    format_answer_for_display,
    SYSTEM_PROMPT_TEMPLATE,
    SIMPLE_PROMPT_TEMPLATE
)


class TestGenerateAnswer:
    """Test cases for main generate_answer function."""
    
    @pytest.fixture
    def sample_chunks(self):
        """Sample chunks for testing."""
        return [
            {
                "id": "doc1_0",
                "text": "Machine learning is a subset of AI that enables systems to learn from data.",
                "metadata": {"source": "ml_guide.txt", "chunk_index": 0},
                "score": 0.85
            },
            {
                "id": "doc1_1",
                "text": "Deep learning uses neural networks with multiple layers for complex pattern recognition.",
                "metadata": {"source": "ml_guide.txt", "chunk_index": 1},
                "score": 0.72
            }
        ]
    
    def test_generate_answer_returns_dict(self, sample_chunks):
        """Test that generate_answer returns a dictionary."""
        result = generate_answer("What is machine learning?", sample_chunks)
        assert isinstance(result, dict)
    
    def test_generate_answer_has_required_keys(self, sample_chunks):
        """Test that result has all required keys."""
        result = generate_answer("What is machine learning?", sample_chunks)
        
        assert 'answer' in result
        assert 'sources' in result
        assert 'confidence' in result
        assert 'method' in result
    
    def test_generate_answer_types(self, sample_chunks):
        """Test that result values have correct types."""
        result = generate_answer("What is machine learning?", sample_chunks)
        
        assert isinstance(result['answer'], str)
        assert isinstance(result['sources'], list)
        assert isinstance(result['confidence'], (int, float))
        assert isinstance(result['method'], str)
    
    def test_generate_answer_confidence_range(self, sample_chunks):
        """Test that confidence is between 0 and 1."""
        result = generate_answer("What is machine learning?", sample_chunks)
        assert 0.0 <= result['confidence'] <= 1.0
    
    def test_generate_answer_with_empty_chunks(self):
        """Test handling of empty chunks list."""
        result = generate_answer("Any question?", [])
        
        assert isinstance(result, dict)
        assert result['confidence'] == 0.0
        assert result['sources'] == []
        assert "couldn't find relevant information" in result['answer'].lower()
        assert result['method'] == 'empty_context'
    
    def test_generate_answer_with_single_chunk(self):
        """Test with single chunk."""
        chunks = [{
            "id": "doc1_0",
            "text": "Python is a programming language.",
            "metadata": {"source": "python.txt", "chunk_index": 0},
            "score": 0.9
        }]
        
        result = generate_answer("What is Python?", chunks)
        assert isinstance(result, dict)
        assert len(result['sources']) > 0


class TestRuleBasedGenerate:
    """Test cases for rule-based fallback generator."""
    
    def test_rule_based_with_empty_chunks(self):
        """Test rule-based generator with no chunks."""
        result = rule_based_generate("Any question?", [])
        
        assert "couldn't find relevant information" in result['answer'].lower()
        assert result['sources'] == []
        assert result['confidence'] == 0.0
    
    def test_rule_based_with_relevant_chunks(self):
        """Test rule-based generator with relevant chunks."""
        chunks = [{
            "id": "doc1_0",
            "text": "Machine learning is a method of data analysis that automates analytical model building.",
            "metadata": {"source": "ml.txt", "chunk_index": 0},
            "score": 0.8
        }]
        
        result = rule_based_generate("What is machine learning?", chunks)
        
        assert isinstance(result['answer'], str)
        assert len(result['answer']) > 0
        assert result['confidence'] > 0
        assert result['method'] == 'fallback'
    
    def test_rule_based_extracts_sources(self):
        """Test that sources are extracted from chunks."""
        chunks = [
            {
                "id": "doc1_0",
                "text": "AI is intelligence demonstrated by machines.",
                "metadata": {"source": "ai_basics.pdf", "chunk_index": 0},
                "score": 0.9
            },
            {
                "id": "doc2_0",
                "text": "Machine learning is a subset of AI.",
                "metadata": {"source": "ml_intro.txt", "chunk_index": 0},
                "score": 0.7
            }
        ]
        
        result = rule_based_generate("What is AI?", chunks)
        assert len(result['sources']) > 0
        assert any('pdf' in s or 'txt' in s for s in result['sources'])


class TestFormatAnswerForDisplay:
    """Test cases for answer formatting."""
    
    def test_format_basic_answer(self):
        """Test formatting of basic answer."""
        result = {
            "answer": "This is the answer.",
            "sources": ["doc1.txt"],
            "confidence": 0.85,
            "method": "llm"
        }
        
        formatted = format_answer_for_display(result)
        
        assert "This is the answer." in formatted
        assert "doc1.txt" in formatted
        assert "85%" in formatted or "0.85" in formatted
    
    def test_format_with_multiple_sources(self):
        """Test formatting with multiple sources."""
        result = {
            "answer": "Answer text",
            "sources": ["doc1.txt", "doc2.pdf", "doc3.docx"],
            "confidence": 0.75,
            "method": "llm"
        }
        
        formatted = format_answer_for_display(result)
        
        assert "doc1.txt" in formatted
        assert "doc2.pdf" in formatted
        assert "doc3.docx" in formatted
    
    def test_format_fallback_method(self):
        """Test that fallback method shows warning."""
        result = {
            "answer": "Answer text",
            "sources": ["doc1.txt"],
            "confidence": 0.5,
            "method": "fallback"
        }
        
        formatted = format_answer_for_display(result)
        assert "fallback" in formatted.lower() or "note" in formatted.lower()
    
    def test_format_with_no_sources(self):
        """Test formatting when no sources available."""
        result = {
            "answer": "No information found.",
            "sources": [],
            "confidence": 0.0,
            "method": "empty_context"
        }
        
        formatted = format_answer_for_display(result)
        assert "No information found." in formatted


class TestPromptTemplates:
    """Test cases for prompt templates."""
    
    def test_system_prompt_template_has_placeholders(self):
        """Test that system prompt has required placeholders."""
        assert "{context}" in SYSTEM_PROMPT_TEMPLATE
        assert "{question}" in SYSTEM_PROMPT_TEMPLATE
    
    def test_simple_prompt_template_has_placeholders(self):
        """Test that simple prompt has required placeholders."""
        assert "{context}" in SIMPLE_PROMPT_TEMPLATE
        assert "{question}" in SIMPLE_PROMPT_TEMPLATE
    
    def test_prompt_template_formatting(self):
        """Test that prompt templates can be formatted."""
        test_context = "Sample context text"
        test_question = "Sample question?"
        
        formatted = SYSTEM_PROMPT_TEMPLATE.format(
            context=test_context,
            question=test_question
        )
        
        assert test_context in formatted
        assert test_question in formatted


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_very_long_query(self):
        """Test with very long query."""
        long_query = "What is " + "machine learning " * 100 + "?"
        chunks = [{
            "id": "doc1_0",
            "text": "Machine learning info",
            "metadata": {"source": "ml.txt", "chunk_index": 0},
            "score": 0.8
        }]
        
        result = generate_answer(long_query, chunks)
        assert isinstance(result, dict)
    
    def test_empty_query(self):
        """Test with empty query string."""
        chunks = [{
            "id": "doc1_0",
            "text": "Some text",
            "metadata": {"source": "doc.txt", "chunk_index": 0},
            "score": 0.8
        }]
        
        result = generate_answer("", chunks)
        assert isinstance(result, dict)
    
    def test_chunks_without_metadata(self):
        """Test chunks with missing metadata."""
        chunks = [{
            "id": "doc1_0",
            "text": "Some text here",
            "metadata": {},  # Empty metadata
            "score": 0.8
        }]
        
        result = generate_answer("Question?", chunks)
        assert isinstance(result, dict)
        # Should handle missing metadata gracefully
    
    def test_chunks_with_special_characters(self):
        """Test chunks containing special characters."""
        chunks = [{
            "id": "doc1_0",
            "text": "Text with special chars: @#$%^&*(){}[]<>~`",
            "metadata": {"source": "special.txt", "chunk_index": 0},
            "score": 0.7
        }]
        
        result = generate_answer("What about special chars?", chunks)
        assert isinstance(result, dict)


class TestBackendSelection:
    """Test LLM backend selection logic."""
    
    def test_generate_with_auto_backend(self):
        """Test automatic backend selection."""
        chunks = [{
            "id": "doc1_0",
            "text": "Sample text",
            "metadata": {"source": "doc.txt", "chunk_index": 0},
            "score": 0.8
        }]
        
        result = generate_answer("Question?", chunks, backend="auto")
        assert isinstance(result, dict)
        # Should fallback to rule-based when no APIs available
        assert result['method'] in ['fallback', 'llm', 'llm_raw']


if __name__ == "__main__":
    """Run tests with pytest."""
    print("=" * 80)
    print("RUNNING GENERATOR UNIT TESTS")
    print("=" * 80)
    print("\nNote: These tests use the rule-based fallback since LLM APIs")
    print("are not configured in the test environment.\n")
    print("-" * 80)
    
    # Run pytest
    pytest.main([__file__, "-v", "--tb=short"])
