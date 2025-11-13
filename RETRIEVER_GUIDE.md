# Retriever Module - Quick Reference

## Overview
The `retriever.py` module provides semantic search with keyword fallback for document retrieval in the RAG chatbot.

## Features
✅ **Semantic Search**: Uses sentence-transformers embeddings for semantic matching  
✅ **Keyword Fallback**: Falls back to substring matching when semantic similarity is low  
✅ **Configurable Threshold**: Adjust similarity threshold for fallback behavior  
✅ **Unit Testable**: Comprehensive test suite included  
✅ **Production Ready**: Error handling, logging, and edge case management  

## Quick Start

### Basic Usage
```python
from retriever import get_top_k

# Simple retrieval
results = get_top_k("your query here", k=4)

# Each result has: id, text, metadata, score
for result in results:
    print(f"Score: {result['score']:.4f}")
    print(f"Text: {result['text'][:100]}...")
    print(f"Source: {result['metadata']['source']}")
```

### Advanced Usage
```python
from retriever import Retriever

# Create retriever with custom settings
retriever = Retriever(
    chroma_db_path="./chroma_db",
    collection_name="capstone_docs",
    embedding_model="all-MiniLM-L6-v2",
    similarity_threshold=0.2  # Adjust for fallback behavior
)

# Get top-k results
results = retriever.get_top_k(query="machine learning", k=5)

# Get collection statistics
stats = retriever.get_collection_stats()
print(f"Total documents: {stats['total_documents']}")
```

## Function Reference

### `get_top_k(query, k=4, ...)`
Convenience function for quick retrieval without managing Retriever instance.

**Parameters:**
- `query` (str): Query string
- `k` (int): Number of results to return (default: 4)
- `chroma_db_path` (str): Path to ChromaDB (default: "./chroma_db")
- `collection_name` (str): Collection name (default: "capstone_docs")
- `similarity_threshold` (float): Minimum avg similarity (default: 0.2)

**Returns:**
- `List[Dict]`: Results with format `{"id": str, "text": str, "metadata": dict, "score": float}`

### `Retriever` Class

#### Constructor
```python
Retriever(
    chroma_db_path="./chroma_db",
    collection_name="capstone_docs", 
    embedding_model="all-MiniLM-L6-v2",
    similarity_threshold=0.2
)
```

#### Methods

**`get_top_k(query, k=4)`**
- Main retrieval method
- Returns semantic search results or keyword fallback
- Format: `[{"id": str, "text": str, "metadata": dict, "score": float}]`

**`get_collection_stats()`**
- Returns collection statistics
- Includes: total documents, embedding model, storage path

## How It Works

### 1. Semantic Search (Primary Method)
- Embeds query using sentence-transformers
- Performs vector similarity search in ChromaDB
- Returns top-k most similar documents
- Calculates similarity scores (0-1 range)

### 2. Keyword Fallback (When Needed)
Triggers when average semantic similarity < threshold (default: 0.2)

- Searches for exact substring matches
- Scores based on keyword frequency
- Returns top-k matches by score

### 3. Decision Logic
```
IF avg_semantic_similarity >= threshold:
    RETURN semantic_results
ELSE:
    TRY keyword_search
    IF keyword_results_found:
        RETURN keyword_results
    ELSE:
        RETURN semantic_results  # Better than nothing
```

## Result Format

```python
{
    "id": "document.txt_0",           # Unique chunk ID
    "text": "chunk content...",        # Full chunk text
    "metadata": {                      # Chunk metadata
        "source": "document.txt",
        "chunk_index": 0
    },
    "score": 0.8542                    # Similarity score (0-1)
}
```

## Similarity Threshold Examples

| Threshold | Behavior |
|-----------|----------|
| 0.1 | Very permissive - almost always uses semantic search |
| 0.2 | **Default** - balanced between semantic and keyword |
| 0.5 | Stricter - more likely to use keyword fallback |
| 0.9 | Very strict - frequently uses keyword fallback |

## Testing

### Run Unit Tests
```bash
python -m pytest test_retriever.py -v
```

### Run Demo
```bash
python retriever.py
```

### Quick Test
```python
from retriever import get_top_k

# Test semantic search
results = get_top_k("artificial intelligence", k=3)
print(f"Found {len(results)} results")

# Test with high threshold (triggers keyword fallback)
from retriever import Retriever
r = Retriever(similarity_threshold=0.9)
results = r.get_top_k("specific keyword", k=2)
```

## Integration with RAG Pipeline

```python
from retriever import get_top_k

def answer_question(question: str) -> str:
    # 1. Retrieve relevant context
    context_chunks = get_top_k(question, k=5)
    
    # 2. Format context
    context = "\n\n".join([chunk['text'] for chunk in context_chunks])
    
    # 3. Generate answer with LLM
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    answer = llm.generate(prompt)
    
    return answer
```

## Performance Tips

1. **Adjust k based on document length**: Longer documents → smaller k
2. **Tune similarity threshold**: Test with your specific documents
3. **Reuse Retriever instance**: Avoid reloading model for each query
4. **Monitor fallback frequency**: High fallback rate may indicate issues

## Troubleshooting

### "ChromaDB path not found"
**Solution:** Run `python embeddings_and_chroma_setup.py` first

### "Collection not found"
**Solution:** Ensure collection name matches the one created during setup

### No results returned
**Possible causes:**
- Empty query
- No documents in collection
- Check collection stats: `retriever.get_collection_stats()`

### Low similarity scores
**Solutions:**
- Lower similarity threshold
- Add more diverse documents
- Check if query matches document domain

## Files
- `retriever.py` - Main retrieval module
- `test_retriever.py` - Unit tests (13 tests)
- `chroma_db/` - ChromaDB persistent storage

## Dependencies
- chromadb
- sentence-transformers
- pytest (for testing)

---
**Status:** ✅ Production Ready  
**Tests:** ✅ 13/13 Passing  
**Last Updated:** 2025-11-13
