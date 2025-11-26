"""
Startup script for Hugging Face Space
"""
import os
import sys
import traceback

print("===== Application Startup =====")
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")

try:
    print("Starting Company Data Q&A System...")

    # Check if ChromaDB exists, if not create it
    if not os.path.exists("chroma_db"):
        print("Building vector database from company documents...")
        import subprocess
        result = subprocess.run([sys.executable, "embeddings_and_chroma_setup.py"], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print("ERROR: Failed to build database")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            sys.exit(1)
        print("Database built successfully!")
    else:
        print("Using existing vector database")

    # Start the app
    print("Launching application...")
    from app_github import main
    main()
    
except Exception as e:
    print(f"FATAL ERROR: {str(e)}")
    traceback.print_exc()
    sys.exit(1)
