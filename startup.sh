#!/bin/bash
# Startup script for Hugging Face Space
# Indexes documents before starting the app

echo "========================================="
echo "RAG Chatbot - Startup"
echo "========================================="

# Check if chroma_db exists
if [ ! -d "chroma_db" ]; then
    echo "📚 ChromaDB not found. Indexing documents..."
    python embeddings_and_chroma_setup.py
    
    if [ $? -eq 0 ]; then
        echo "✅ Documents indexed successfully"
    else
        echo "❌ Failed to index documents"
        exit 1
    fi
else
    echo "✅ ChromaDB found. Skipping indexing."
fi

echo "🚀 Starting RAG application..."
python app_github.py
