"""
Simple test to check if basic imports work
"""
print("Testing RAG Chatbot setup...")
print("-" * 50)

# Test 1: Check Python version
import sys
print(f"✓ Python version: {sys.version}")

# Test 2: Try importing each dependency
dependencies = {
    "gradio": "Gradio UI framework",
    "chromadb": "Vector database",
    "sentence_transformers": "Embedding model",
    "dotenv": "Environment variables",
    "PyPDF2": "PDF processing",
    "docx": "DOCX processing",
    "openai": "OpenAI API"
}

print("\nChecking dependencies:")
print("-" * 50)

missing = []
for module, description in dependencies.items():
    try:
        __import__(module)
        print(f"✓ {module:25} - {description}")
    except ImportError:
        print(f"✗ {module:25} - NOT INSTALLED")
        missing.append(module)

print("-" * 50)

if missing:
    print(f"\n⚠️  Missing {len(missing)} package(s): {', '.join(missing)}")
    print("\nTo install, run:")
    print(f"pip install {' '.join(missing)}")
else:
    print("\n✅ All dependencies installed!")
    print("\nYou can now run: python app.py")
