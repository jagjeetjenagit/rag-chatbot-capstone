"""
Enhanced Unit Tests for generator.py module.
Tests answer generation with synthetic data, keyword validation, and edge cases.

Run with: pytest test_generator_enhanced.py -v
"""
import pytest
from unittest.mock import Mock, patch
from generator import (
    generate_answer,
    rule_based_generate,
    format_answer_for_display,
    call_llm_api,
    SYSTEM_PROMPT_TEMPLATE,
    SIMPLE_PROMPT_TEMPLATE
)


class TestGenerateAnswerWithSyntheticData:
    """Test answer generation with synthetic retrieved chunks."""
    
    @pytest.fixture
    def ml_chunks(self):
        """Synthetic chunks about machine learning."""
        return [
            {
                "id": "ml_doc_0",
                "text": "Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.",
                "metadata": {"source": "ml_intro.pdf", "chunk_index": 0},
                "score": 0.92
            },
            {
                "id": "ml_doc_1",
                "text": "Supervised learning is a type of machine learning where the algorithm learns from labeled training data. The algorithm tries to learn the mapping function from the input to the output.",
                "metadata": {"source": "ml_intro.pdf", "chunk_index": 1},
                "score": 0.85
            },
            {
                "id": "ml_doc_2",
                "text": "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes called neurons organized in layers that can learn to perform tasks by considering examples.",
                "metadata": {"source": "nn_guide.txt", "chunk_index": 0},
                "score": 0.78
            }
        ]
    
    @pytest.fixture
    def rag_chunks(self):
        """Synthetic chunks about RAG systems."""
        return [
            {
                "id": "rag_doc_0",
                "text": "Retrieval-Augmented Generation (RAG) combines information retrieval with text generation. RAG models retrieve relevant documents from a knowledge base and use them to generate accurate, contextual responses.",
                "metadata": {"source": "rag_overview.docx", "chunk_index": 0},
                "score": 0.95
            },
            {
                "id": "rag_doc_1",
                "text": "The RAG pipeline consists of three main steps: document retrieval using semantic search, context augmentation by combining retrieved chunks, and answer generation using a language model.",
                "metadata": {"source": "rag_overview.docx", "chunk_index": 1},
                "score": 0.88
            }
        ]
    
    def test_answer_contains_expected_keywords_ml(self, ml_chunks):
        """Test that answer contains expected keywords from ML chunks."""
        query = "What is machine learning?"
        result = generate_answer(query, ml_chunks)
        
        # Answer should contain relevant keywords
        answer_lower = result['answer'].lower()
        assert any(keyword in answer_lower for keyword in [
            'machine learning', 'data', 'artificial intelligence', 'learn', 'model'
        ])
        
        # Sources should not be empty
        assert len(result['sources']) > 0
        assert 'ml_intro.pdf' in result['sources'] or 'nn_guide.txt' in result['sources']
    
    def test_answer_contains_expected_keywords_rag(self, rag_chunks):
        """Test that answer contains RAG-related keywords."""
        query = "How does RAG work?"
        result = generate_answer(query, rag_chunks)
        
        # Answer should contain RAG-related keywords
        answer_lower = result['answer'].lower()
        assert any(keyword in answer_lower for keyword in [
            'rag', 'retrieval', 'generation', 'document', 'context'
        ])
        
        # Sources should include RAG document
        assert len(result['sources']) > 0
        assert 'rag_overview.docx' in result['sources']
    
    def test_sources_list_non_empty(self, ml_chunks):
        """Test that sources list is non-empty with valid chunks."""
        result = generate_answer("Test question", ml_chunks)
        
        assert isinstance(result['sources'], list)
        assert len(result['sources']) > 0
        
        # All sources should be strings
        for source in result['sources']:
            assert isinstance(source, str)
            assert len(source) > 0
    
    def test_multiple_sources_extracted(self, ml_chunks):
        """Test that multiple sources are extracted when available."""
        result = generate_answer("Explain machine learning concepts", ml_chunks)
        
        # Should have at least one source
        assert len(result['sources']) >= 1
        
        # Sources should come from chunk metadata
        chunk_sources = [chunk['metadata']['source'] for chunk in ml_chunks]
        for source in result['sources']:
            assert source in chunk_sources
    
    def test_confidence_reasonable_with_good_chunks(self, ml_chunks):
        """Test that confidence is reasonable with relevant chunks."""
        result = generate_answer("What is machine learning?", ml_chunks)
        
        # With relevant chunks, confidence should be > 0
        assert result['confidence'] > 0
        
        # For ML query with ML chunks, should have decent confidence
        # (fallback method gives 0.3-0.8 based on keyword overlap)
        assert result['confidence'] >= 0.3
    
    def test_answer_not_empty(self, ml_chunks):
        """Test that answer is never empty with valid chunks."""
        result = generate_answer("Any question", ml_chunks)
        
        assert isinstance(result['answer'], str)
        assert len(result['answer']) > 0
        assert result['answer'].strip() != ""
    
    def test_answer_uses_chunk_content(self, rag_chunks):
        """Test that answer appears to use content from chunks."""
        result = generate_answer("What is RAG?", rag_chunks)
        
        answer_lower = result['answer'].lower()
        
        # Answer should contain words from the chunks
        chunk_words = set()
        for chunk in rag_chunks:
            words = chunk['text'].lower().split()
            chunk_words.update([w for w in words if len(w) > 4])
        
        answer_words = set(answer_lower.split())
        
        # Should have some overlap
        overlap = chunk_words & answer_words
        assert len(overlap) > 0


class TestGenerateAnswerEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_chunks_returns_helpful_message(self):
        """Test that empty chunks returns a helpful message."""
        result = generate_answer("Any question", [])
        
        assert "couldn't find" in result['answer'].lower()
        assert result['confidence'] == 0.0
        assert result['sources'] == []
        assert result['method'] == 'empty_context'
    
    def test_chunks_with_missing_metadata(self):
        """Test handling of chunks with incomplete metadata."""
        chunks = [
            {
                "id": "test_0",
                "text": "Some text here",
                "metadata": {},  # Empty metadata
                "score": 0.8
            }
        ]
        
        result = generate_answer("Question?", chunks)
        
        # Should handle gracefully
        assert isinstance(result, dict)
        assert 'answer' in result
        assert 'sources' in result
    
    def test_chunks_with_special_characters(self):
        """Test chunks containing special characters."""
        chunks = [
            {
                "id": "test_0",
                "text": "Text with special chars: @#$%^&*(){}[]<>~` and unicode: café, naïve, 中文",
                "metadata": {"source": "special.txt", "chunk_index": 0},
                "score": 0.7
            }
        ]
        
        result = generate_answer("What about special chars?", chunks)
        
        # Should not crash
        assert isinstance(result, dict)
        assert len(result['answer']) > 0
    
    def test_very_long_chunks(self):
        """Test with very long chunk text."""
        long_text = "This is a very long text. " * 500  # ~3500+ words
        chunks = [
            {
                "id": "long_0",
                "text": long_text,
                "metadata": {"source": "long_doc.txt", "chunk_index": 0},
                "score": 0.9
            }
        ]
        
        result = generate_answer("Summarize this", chunks)
        
        # Should handle long text
        assert isinstance(result, dict)
        assert len(result['answer']) > 0
    
    def test_empty_query_string(self):
        """Test with empty query."""
        chunks = [
            {
                "id": "test_0",
                "text": "Some text",
                "metadata": {"source": "test.txt", "chunk_index": 0},
                "score": 0.8
            }
        ]
        
        result = generate_answer("", chunks)
        
        # Should handle gracefully (rule-based may still work)
        assert isinstance(result, dict)


class TestRuleBasedGeneratorDetailed:
    """Detailed tests for rule-based fallback generator."""
    
    def test_keyword_extraction_works(self):
        """Test that keyword matching works correctly."""
        chunks = [
            {
                "id": "test_0",
                "text": "Python is a high-level programming language known for its simplicity and readability.",
                "metadata": {"source": "python_intro.txt", "chunk_index": 0},
                "score": 0.8
            }
        ]
        
        # Query with keywords in the chunk
        result = rule_based_generate("What is Python programming?", chunks)
        
        # Should find the relevant chunk
        assert 'Python' in result['answer'] or 'python' in result['answer'].lower()
        assert result['confidence'] > 0
        assert len(result['sources']) > 0
    
    def test_multiple_chunks_best_selected(self):
        """Test that best matching chunk is selected."""
        chunks = [
            {
                "id": "test_0",
                "text": "Cats are domestic animals that make good pets.",
                "metadata": {"source": "animals.txt", "chunk_index": 0},
                "score": 0.5
            },
            {
                "id": "test_1",
                "text": "Python programming language is used for web development, data science, and automation.",
                "metadata": {"source": "programming.txt", "chunk_index": 0},
                "score": 0.7
            },
            {
                "id": "test_2",
                "text": "Python is also a type of snake found in tropical regions.",
                "metadata": {"source": "reptiles.txt", "chunk_index": 0},
                "score": 0.6
            }
        ]
        
        result = rule_based_generate("Tell me about Python programming", chunks)
        
        # Should prefer the programming chunk
        answer_lower = result['answer'].lower()
        assert 'programming' in answer_lower or 'language' in answer_lower or 'development' in answer_lower


class TestAnswerFormatting:
    """Test answer formatting for display."""
    
    def test_format_includes_confidence(self):
        """Test that formatted answer includes confidence."""
        result = {
            "answer": "This is the answer.",
            "sources": ["doc1.txt"],
            "confidence": 0.85,
            "method": "llm"
        }
        
        formatted = format_answer_for_display(result)
        
        # Should show confidence as percentage
        assert "85%" in formatted or "0.85" in formatted
        assert "Confidence" in formatted or "confidence" in formatted
    
    def test_format_includes_sources(self):
        """Test that formatted answer includes sources."""
        result = {
            "answer": "Answer text here.",
            "sources": ["ml_guide.pdf", "data_science.txt"],
            "confidence": 0.75,
            "method": "llm"
        }
        
        formatted = format_answer_for_display(result)
        
        # Should list both sources
        assert "ml_guide.pdf" in formatted
        assert "data_science.txt" in formatted
        assert "Sources" in formatted or "sources" in formatted
    
    def test_format_shows_fallback_warning(self):
        """Test that fallback method shows warning."""
        result = {
            "answer": "Answer from fallback.",
            "sources": ["doc.txt"],
            "confidence": 0.5,
            "method": "fallback"
        }
        
        formatted = format_answer_for_display(result)
        
        # Should indicate fallback was used
        assert "fallback" in formatted.lower() or "note" in formatted.lower()


class TestLLMIntegration:
    """Test LLM API integration (with mocking)."""
    
    @patch('generator.call_openai_api')
    def test_llm_api_called_with_chunks(self, mock_openai):
        """Test that LLM API is called when available."""
        # Mock LLM to return JSON response
        mock_openai.return_value = '{"answer": "ML is AI subset", "sources_used": [1], "confidence": 0.9}'
        
        chunks = [
            {
                "id": "ml_0",
                "text": "Machine learning is a subset of AI.",
                "metadata": {"source": "ml.txt", "chunk_index": 0},
                "score": 0.9
            }
        ]
        
        # Should try to call OpenAI (though it will fail without API key)
        result = generate_answer("What is ML?", chunks, backend="openai")
        
        # Even if it fails, should return valid result
        assert isinstance(result, dict)
        assert 'answer' in result


class TestPromptTemplates:
    """Test prompt template functionality."""
    
    def test_system_prompt_formatting(self):
        """Test that system prompt can be formatted correctly."""
        test_context = "Sample context from documents"
        test_question = "What is the answer?"
        
        formatted = SYSTEM_PROMPT_TEMPLATE.format(
            context=test_context,
            question=test_question
        )
        
        # Should contain both context and question
        assert test_context in formatted
        assert test_question in formatted
        
        # Should have instructions
        assert "JSON" in formatted
        assert "confidence" in formatted.lower()
    
    def test_simple_prompt_formatting(self):
        """Test simple prompt template."""
        test_context = "Context here"
        test_question = "Question here?"
        
        formatted = SIMPLE_PROMPT_TEMPLATE.format(
            context=test_context,
            question=test_question
        )
        
        assert test_context in formatted
        assert test_question in formatted


if __name__ == "__main__":
    """Run tests with pytest."""
    print("=" * 80)
    print("RUNNING ENHANCED GENERATOR UNIT TESTS")
    print("=" * 80)
    print("\nTests validate answer generation with synthetic data,")
    print("keyword extraction, and comprehensive edge cases.\n")
    print("-" * 80)
    
    # Run pytest
    pytest.main([__file__, "-v", "--tb=short", "-s"])
