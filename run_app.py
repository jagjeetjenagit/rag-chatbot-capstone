"""
Startup script for Hugging Face Space
Indexes documents before starting the RAG application
"""
import os
import sys
import subprocess
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run document indexing, then start the app"""
    
    print("=" * 80)
    print("RAG Chatbot - Company Data Q&A System")
    print("=" * 80)
    
    # Log Python version and working directory
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Directory contents: {os.listdir('.')}")
    
    # Check for data folder
    data_path = "data/documents/company_data"
    if os.path.exists(data_path):
        files = os.listdir(data_path)
        logger.info(f"✅ Found {len(files)} files in {data_path}")
        logger.info(f"Files: {files[:3]}...")  # Show first 3 files
    else:
        logger.error(f"❌ Data path not found: {data_path}")
        logger.info("Available paths:")
        if os.path.exists("data"):
            logger.info(f"  data/: {os.listdir('data')}")
            if os.path.exists("data/documents"):
                logger.info(f"  data/documents/: {os.listdir('data/documents')}")
    
    # Check if chroma_db exists
    if not os.path.exists("chroma_db"):
        logger.info("📚 ChromaDB not found. Indexing documents...")
        
        # Run the indexing script
        result = subprocess.run(
            [sys.executable, "embeddings_and_chroma_setup.py"],
            capture_output=True,
            text=True
        )
        
        logger.info("STDOUT:")
        logger.info(result.stdout)
        
        if result.stderr:
            logger.warning("STDERR:")
            logger.warning(result.stderr)
        
        if result.returncode == 0:
            logger.info("✅ Documents indexed successfully")
            
            # Verify ChromaDB was created
            if not os.path.exists("chroma_db"):
                logger.error("❌ ChromaDB directory not created after indexing!")
                sys.exit(1)
            
            # Check if collection has documents
            try:
                import chromadb
                from chroma_client import get_chroma_client
                client = get_chroma_client("./chroma_db")
                collection = client.get_collection("capstone_docs")
                count = collection.count()
                logger.info(f"✅ ChromaDB collection has {count} documents")
                if count == 0:
                    logger.error("❌ ChromaDB collection is empty!")
                    sys.exit(1)
            except Exception as e:
                logger.error(f"❌ Failed to verify ChromaDB: {e}")
                sys.exit(1)
        else:
            logger.error("❌ Failed to index documents")
            logger.error(f"Return code: {result.returncode}")
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
