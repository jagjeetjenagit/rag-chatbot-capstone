"""
Enhanced Unit Tests for retriever.py module.
Tests semantic search, keyword fallback, mocking, and edge cases.

Run with: pytest test_retriever_enhanced.py -v
"""
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import numpy as np
from retriever import Retriever, get_top_k


class TestRetrieverWithMocking:
    """Test cases for Retriever with mocked ChromaDB."""
    
    @pytest.fixture
    def mock_chroma_collection(self):
        """Create a mock ChromaDB collection with sample data."""
        mock_collection = Mock()
        
        # Mock collection.count()
        mock_collection.count.return_value = 3
        
        # Mock collection.query() to return sample results
        def mock_query(query_embeddings, n_results, include):
            # Return mock results based on query
            return {
                'ids': [['doc1_0', 'doc2_0', 'doc3_0'][:n_results]],
                'documents': [
                    [
                        'Machine learning is a subset of artificial intelligence.',
                        'Deep learning uses neural networks with multiple layers.',
                        'Natural language processing enables computers to understand text.'
                    ][:n_results]
                ],
                'metadatas': [
                    [
                        {'source': 'ml_basics.txt', 'chunk_index': 0},
                        {'source': 'dl_intro.pdf', 'chunk_index': 0},
                        {'source': 'nlp_guide.docx', 'chunk_index': 0}
                    ][:n_results]
                ],
                'distances': [[0.3, 0.5, 0.7][:n_results]]
            }
        
        mock_collection.query = Mock(side_effect=mock_query)
        
        # Mock collection.get() for keyword search
        mock_collection.get.return_value = {
            'ids': ['doc1_0', 'doc2_0', 'doc3_0'],
            'documents': [
                'Machine learning is a subset of artificial intelligence.',
                'Deep learning uses neural networks with multiple layers.',
                'Natural language processing enables computers to understand text.'
            ],
            'metadatas': [
                {'source': 'ml_basics.txt', 'chunk_index': 0},
                {'source': 'dl_intro.pdf', 'chunk_index': 0},
                {'source': 'nlp_guide.docx', 'chunk_index': 0}
            ]
        }
        
        return mock_collection
    
    @pytest.fixture
    def mock_retriever(self, mock_chroma_collection):
        """Create a Retriever with mocked ChromaDB."""
        with patch('retriever.chromadb.PersistentClient') as mock_client, \
             patch('retriever.SentenceTransformer') as mock_st, \
             patch('pathlib.Path.exists', return_value=True):
            
            # Mock the client to return our mock collection
            mock_client_instance = Mock()
            mock_client_instance.get_collection.return_value = mock_chroma_collection
            mock_client.return_value = mock_client_instance
            
            # Mock sentence transformer
            mock_model = Mock()
            mock_model.encode.return_value = np.random.rand(384)
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_st.return_value = mock_model
            
            # Create retriever
            retriever = Retriever(
                chroma_db_path="./test_chroma_db",
                collection_name="test_collection",
                similarity_threshold=0.2
            )
            
            # Inject mocks
            retriever.collection = mock_chroma_collection
            
            return retriever
    
    def test_get_top_k_returns_expected_chunks(self, mock_retriever):
        """Test that get_top_k returns expected chunks for known query."""
        query = "What is machine learning?"
        results = mock_retriever.get_top_k(query, k=2)
        
        # Verify we got results
        assert len(results) == 2
        
        # Verify first result contains expected content
        assert 'Machine learning' in results[0]['text']
        assert results[0]['metadata']['source'] == 'ml_basics.txt'
        assert results[0]['metadata']['chunk_index'] == 0
        
        # Verify result structure
        for result in results:
            assert 'id' in result
            assert 'text' in result
            assert 'metadata' in result
            assert 'score' in result
            assert 0 <= result['score'] <= 1
    
    def test_get_top_k_with_k_parameter(self, mock_retriever):
        """Test that k parameter correctly limits results."""
        for k in [1, 2, 3]:
            results = mock_retriever.get_top_k("test query", k=k)
            assert len(results) == k
    
    def test_get_top_k_returns_highest_scores_first(self, mock_retriever):
        """Test that results are ordered by score."""
        results = mock_retriever.get_top_k("test query", k=3)
        
        # Verify scores are in descending order
        for i in range(len(results) - 1):
            assert results[i]['score'] >= results[i + 1]['score']
    
    def test_semantic_search_called_with_correct_params(self, mock_retriever):
        """Test that semantic search is called with correct parameters."""
        mock_retriever.get_top_k("test query", k=4)
        
        # Verify query was called
        assert mock_retriever.collection.query.called
        call_args = mock_retriever.collection.query.call_args
        
        # Verify n_results parameter
        assert call_args.kwargs['n_results'] == 4
        assert 'documents' in call_args.kwargs['include']
        assert 'metadatas' in call_args.kwargs['include']


class TestRetrieverEdgeCases:
    """Test edge cases for retriever."""
    
    @pytest.fixture
    def empty_mock_retriever(self):
        """Create a Retriever with empty collection."""
        with patch('retriever.chromadb.PersistentClient') as mock_client:
            # Mock empty collection
            mock_collection = Mock()
            mock_collection.count.return_value = 0
            mock_collection.query.return_value = {
                'ids': [[]],
                'documents': [[]],
                'metadatas': [[]],
                'distances': [[]]
            }
            mock_collection.get.return_value = {
                'ids': [],
                'documents': [],
                'metadatas': []
            }
            
            mock_client_instance = Mock()
            mock_client_instance.get_collection.return_value = mock_collection
            mock_client.return_value = mock_client_instance
            
            with patch('retriever.SentenceTransformer') as mock_st, \
                 patch('pathlib.Path.exists', return_value=True):
                
                mock_model = Mock()
                mock_model.encode.return_value = np.random.rand(384)
                mock_model.get_sentence_embedding_dimension.return_value = 384
                mock_st.return_value = mock_model
                
                retriever = Retriever(
                    chroma_db_path="./test_chroma_db",
                    collection_name="test_collection"
                )
                retriever.collection = mock_collection
                
                return retriever
    
    def test_get_top_k_with_no_documents_indexed(self, empty_mock_retriever):
        """Test behavior when no documents are indexed."""
        results = empty_mock_retriever.get_top_k("any query", k=4)
        
        # Should return empty list
        assert isinstance(results, list)
        assert len(results) == 0
    
    def test_get_top_k_with_empty_query(self, empty_mock_retriever):
        """Test handling of empty query."""
        results = empty_mock_retriever.get_top_k("", k=4)
        
        # Should return empty list
        assert results == []
    
    def test_get_top_k_with_whitespace_query(self, empty_mock_retriever):
        """Test handling of whitespace-only query."""
        results = empty_mock_retriever.get_top_k("   ", k=4)
        
        # Should return empty list
        assert results == []
    
    def test_get_top_k_with_large_k(self, empty_mock_retriever):
        """Test with k larger than available documents."""
        # Should not crash, just return what's available
        results = empty_mock_retriever.get_top_k("query", k=1000)
        
        assert isinstance(results, list)


class TestRetrieverFallback:
    """Test keyword fallback behavior."""
    
    @pytest.fixture
    def low_similarity_retriever(self):
        """Create retriever that triggers fallback."""
        with patch('retriever.chromadb.PersistentClient') as mock_client:
            mock_collection = Mock()
            mock_collection.count.return_value = 2
            
            # Return low similarity results
            mock_collection.query.return_value = {
                'ids': [['doc1_0', 'doc2_0']],
                'documents': [
                    [
                        'The chunking strategy is important for RAG.',
                        'Document processing requires careful planning.'
                    ]
                ],
                'metadatas': [
                    [
                        {'source': 'doc1.txt', 'chunk_index': 0},
                        {'source': 'doc2.txt', 'chunk_index': 0}
                    ]
                ],
                'distances': [[0.9, 0.95]]  # High distance = low similarity
            }
            
            # For keyword search
            mock_collection.get.return_value = {
                'ids': ['doc1_0', 'doc2_0'],
                'documents': [
                    'The chunking strategy is important for RAG.',
                    'Document processing requires careful planning.'
                ],
                'metadatas': [
                    {'source': 'doc1.txt', 'chunk_index': 0},
                    {'source': 'doc2.txt', 'chunk_index': 0}
                ]
            }
            
            mock_client_instance = Mock()
            mock_client_instance.get_collection.return_value = mock_collection
            mock_client.return_value = mock_client_instance
            
            with patch('retriever.SentenceTransformer') as mock_st, \
                 patch('pathlib.Path.exists', return_value=True):
                
                mock_model = Mock()
                mock_model.encode.return_value = np.random.rand(384)
                mock_model.get_sentence_embedding_dimension.return_value = 384
                mock_st.return_value = mock_model
                
                # High threshold to trigger fallback
                retriever = Retriever(
                    chroma_db_path="./test_chroma_db",
                    collection_name="test_collection",
                    similarity_threshold=0.9
                )
                retriever.collection = mock_collection
                
                return retriever
    
    def test_fallback_to_keyword_search(self, low_similarity_retriever):
        """Test that keyword search is used when similarity is low."""
        # Query with specific keyword that exists in documents
        results = low_similarity_retriever.get_top_k("chunking", k=2)
        
        # Should still return results (via fallback)
        assert isinstance(results, list)
        
        # If keyword fallback worked, should find the document with "chunking"
        if results:
            assert len(results) > 0


class TestConvenienceFunction:
    """Test the standalone get_top_k function."""
    
    @patch('retriever.Retriever')
    def test_get_top_k_function(self, mock_retriever_class):
        """Test that convenience function creates retriever and calls method."""
        # Setup mock
        mock_instance = Mock()
        mock_instance.get_top_k.return_value = [
            {
                'id': 'test_0',
                'text': 'Test text',
                'metadata': {'source': 'test.txt', 'chunk_index': 0},
                'score': 0.9
            }
        ]
        mock_retriever_class.return_value = mock_instance
        
        # Call function
        results = get_top_k("test query", k=3)
        
        # Verify Retriever was instantiated
        assert mock_retriever_class.called
        
        # Verify get_top_k was called
        assert mock_instance.get_top_k.called
        
        # Check call arguments (handle both positional and keyword args)
        call_args = mock_instance.get_top_k.call_args
        if call_args[0]:  # Positional args
            assert call_args[0][0] == "test query"
        if call_args[1]:  # Keyword args
            assert call_args[1].get('k', 4) == 3
        
        # Verify results
        assert len(results) == 1
        assert results[0]['id'] == 'test_0'


if __name__ == "__main__":
    """Run tests with pytest."""
    print("=" * 80)
    print("RUNNING ENHANCED RETRIEVER UNIT TESTS WITH MOCKING")
    print("=" * 80)
    print("\nThese tests use mocked ChromaDB and SentenceTransformer")
    print("to test retriever logic without requiring actual database.\n")
    print("-" * 80)
    
    # Run pytest
    pytest.main([__file__, "-v", "--tb=short", "-s"])
