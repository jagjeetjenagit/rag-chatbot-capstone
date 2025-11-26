"""
Startup script for Hugging Face Space
"""
import os
import sys

print("Starting Company Data Q&A System...")

# Check if ChromaDB exists, if not create it
if not os.path.exists("chroma_db"):
    print("Building vector database from company documents...")
    import subprocess
    result = subprocess.run([sys.executable, "embeddings_and_chroma_setup.py"])
    if result.returncode != 0:
        print("ERROR: Failed to build database")
        sys.exit(1)
    print("Database built successfully!")
else:
    print("Using existing vector database")

# Start the app
print("Launching application...")
from app_github import main
main()
