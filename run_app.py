"""
Startup script for Hugging Face Space
Indexes documents before starting the RAG application
"""
import os
import sys
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Run document indexing, then start the app"""
    
    print("=" * 80)
    print("RAG Chatbot - Startup")
    print("=" * 80)
    
    # Check if chroma_db exists
    if not os.path.exists("chroma_db"):
        logger.info("📚 ChromaDB not found. Indexing documents...")
        
        # Run the indexing script
        result = subprocess.run(
            [sys.executable, "embeddings_and_chroma_setup.py"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.info("✅ Documents indexed successfully")
        else:
            logger.error("❌ Failed to index documents")
            logger.error(result.stderr)
            sys.exit(1)
    else:
        logger.info("✅ ChromaDB found. Skipping indexing.")
    
    # Start the RAG application
    logger.info("🚀 Starting RAG application...")
    
    # Import and run the app
    from app_github import main as app_main
    app_main()

if __name__ == "__main__":
    main()
