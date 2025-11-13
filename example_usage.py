"""
Example usage script demonstrating the RAG Chatbot functionality.

This script shows how to use the RAG Chatbot programmatically without the UI.
"""

import os
from pathlib import Path

# Import RAG Chatbot components
from src.document_ingestion import DocumentIngestor, ingest_documents
from src.text_chunking import TextChunker, chunk_documents
from src.vector_store import VectorStore
from src.rag_engine import RAGEngine


def main():
    """Main example function demonstrating RAG pipeline."""
    
    print("=" * 60)
    print("RAG Chatbot - Example Usage")
    print("=" * 60)
    print()
    
    # ===================================================================
    # Step 1: Document Ingestion
    # ===================================================================
    print("Step 1: Ingesting documents...")
    print("-" * 60)
    
    # Initialize the document ingestor
    ingestor = DocumentIngestor()
    
    # Ingest the sample document
    sample_doc = Path("data/sample_document.txt")
    
    if sample_doc.exists():
        text, metadata = ingestor.ingest_document(sample_doc)
        print(f"✓ Ingested: {metadata['source']}")
        print(f"  - File type: {metadata['file_type']}")
        print(f"  - Characters: {metadata['char_count']}")
        print(f"  - Preview: {text[:100]}...")
    else:
        print("⚠️  Sample document not found. Please add documents to data/ folder.")
        return
    
    print()
    
    # ===================================================================
    # Step 2: Text Chunking
    # ===================================================================
    print("Step 2: Chunking text...")
    print("-" * 60)
    
    # Initialize the text chunker
    chunker = TextChunker(
        chunk_size_min=500,
        chunk_size_max=800,
        overlap_percent=10
    )
    
    # Chunk the document
    chunks = chunker.chunk_text(text, metadata["source"], metadata)
    
    print(f"✓ Created {len(chunks)} chunks")
    print(f"  - Chunk size range: {chunker.chunk_size_min}-{chunker.chunk_size_max} chars")
    print(f"  - Overlap: {chunker.overlap_percent}%")
    
    # Show first chunk as example
    if chunks:
        print(f"\n  Example chunk:")
        print(f"  Source: {chunks[0]['metadata']['source']}")
        print(f"  Index: {chunks[0]['metadata']['chunk_index']}")
        print(f"  Text preview: {chunks[0]['text'][:150]}...")
    
    print()
    
    # ===================================================================
    # Step 3: Vector Store
    # ===================================================================
    print("Step 3: Storing in vector database...")
    print("-" * 60)
    
    # Initialize vector store
    vector_store = VectorStore(
        collection_name="example_collection",
        persist_directory="chroma_db"
    )
    
    # Add chunks to vector store
    added_count = vector_store.add_chunks(chunks)
    
    print(f"✓ Added {added_count} chunks to vector database")
    
    # Get database statistics
    stats = vector_store.get_stats()
    print(f"  - Total chunks in database: {stats['total_chunks']}")
    print(f"  - Unique documents: {stats['unique_sources']}")
    print(f"  - Embedding model: {stats['embedding_model']}")
    
    print()
    
    # ===================================================================
    # Step 4: Semantic Search
    # ===================================================================
    print("Step 4: Testing semantic search...")
    print("-" * 60)
    
    # Perform a search
    query = "What is machine learning?"
    print(f"Query: '{query}'")
    print()
    
    results = vector_store.search(query, top_k=3)
    
    print(f"✓ Found {len(results)} relevant chunks:")
    print()
    
    for i, result in enumerate(results, 1):
        print(f"  Result {i}:")
        print(f"  - Source: {result['metadata']['source']}")
        print(f"  - Chunk: {result['metadata']['chunk_index']}")
        print(f"  - Similarity: {result['similarity_score']:.3f}")
        print(f"  - Text: {result['text'][:100]}...")
        print()
    
    # ===================================================================
    # Step 5: RAG Answer Generation
    # ===================================================================
    print("Step 5: Generating answer with RAG...")
    print("-" * 60)
    
    # Check if API key is configured
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
        print("⚠️  No API key found in environment variables.")
        print("   Please set OPENAI_API_KEY or GOOGLE_API_KEY in your .env file")
        print("   to test answer generation.")
        print()
        print("Skipping answer generation step...")
    else:
        try:
            # Initialize RAG engine
            rag_engine = RAGEngine(
                vector_store=vector_store,
                llm_provider=os.getenv("LLM_PROVIDER", "openai")
            )
            
            # Generate answer
            print(f"Query: '{query}'")
            print()
            
            result = rag_engine.generate_answer(query)
            
            if result["context_found"]:
                print("✓ Generated answer:")
                print(f"\n{result['answer']}\n")
                
                print("Sources used:")
                for source in result["sources"]:
                    print(f"  - {source['name']} (similarity: {source['similarity_score']:.3f})")
            else:
                print("⚠️  No relevant context found")
                
        except Exception as e:
            print(f"⚠️  Error generating answer: {str(e)}")
            print("   Make sure your API key is valid and you have credits.")
    
    print()
    
    # ===================================================================
    # Additional Examples
    # ===================================================================
    print("=" * 60)
    print("Additional Features")
    print("=" * 60)
    print()
    
    # Example: Get all sources
    print("All documents in database:")
    sources = vector_store.get_all_sources()
    for source in sources:
        print(f"  - {source}")
    
    print()
    
    # Example: Search with filter
    print("Search with source filter:")
    filtered_results = vector_store.search(
        "artificial intelligence",
        top_k=2,
        filter_metadata={"source": metadata["source"]}
    )
    print(f"  Found {len(filtered_results)} results from {metadata['source']}")
    
    print()
    
    # Clean up option (commented out by default)
    # print("Cleaning up example database...")
    # vector_store.clear()
    # print("✓ Database cleared")
    
    print("=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Add your own documents to the data/ folder")
    print("  2. Run the Gradio UI: python app.py")
    print("  3. Open http://127.0.0.1:7860 in your browser")
    print()


if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Run the example
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExample interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
