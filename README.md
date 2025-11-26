# 📚 RAG Chatbot - Intelligent Document Q&A System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Gradio](https://img.shields.io/badge/Gradio-3.50.2-orange.svg)](https://gradio.app/)

A production-ready **Retrieval-Augmented Generation (RAG)** chatbot that answers questions based on your document collection. Built with ChromaDB, Sentence Transformers, and Gradio. Pre-loaded with 20+ comprehensive documents covering technical topics and business data.

## ⚡ Quick Start (3 Steps)

### 1️⃣ Install Dependencies
```bash
# Activate virtual environment
.venv\Scripts\Activate.ps1

# Install packages
pip install -r requirements.txt
```

### 2️⃣ Add Your API Key
**Only edit the `.env` file - that's where all API keys go!**

Open `.env` and add your key (choose ONE):

```env
# Option A: OpenAI (Recommended)
OPENAI_API_KEY=sk-your-key-here

# Option B: Google Gemini  
GOOGLE_API_KEY=your-key-here

# Option C: Skip this - works without API!
```

**Get API keys:**
- OpenAI: https://platform.openai.com/api-keys (~$0.002 per 1K tokens)
- Google: https://makersuite.google.com/app/apikey (free tier available)

### 3️⃣ Run the App
```bash
# Make sure venv is activated, then:
python start.py

# Or run directly with venv python:
& "C:/capstone project 1/.venv/Scripts/python.exe" start.py
```

Open http://localhost:7860 - Done! 🎉

---

## 🌟 Features

- **Multi-Format Document Support**: Upload PDF, TXT, and DOCX files
- **Smart Text Chunking**: Automatic text splitting (500-800 characters) with 10% overlap for optimal retrieval
- **Semantic Search**: Uses sentence-transformers (all-MiniLM-L6-v2) for high-quality embeddings
- **Vector Storage**: ChromaDB for efficient similarity search
- **LLM Integration**: Supports OpenAI GPT and Google Gemini models
- **Source Attribution**: Answers include citations to source documents
- **Interactive UI**: Clean Gradio interface with file upload and chat capabilities
- **Session Management**: Maintains conversation history
- **Comprehensive Testing**: Unit and integration tests included

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [API Reference](#api-reference)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) Virtual environment tool

### Step 1: Clone or Download the Project

```bash
cd "capstone project 1"
```

### Step 2: Create Virtual Environment (Recommended)

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages:
- `chromadb` - Vector database
- `sentence-transformers` - Embedding generation
- `gradio` - Web UI framework
- `PyPDF2` - PDF processing
- `python-docx` - DOCX processing
- `openai` / `google-generativeai` - LLM APIs
- And more...

### Step 4: Configure API Keys

1. Copy the example environment file:
   ```powershell
   Copy-Item .env.example .env
   ```

2. Edit `.env` and add your API key:
   ```env
   LLM_PROVIDER=openai
   OPENAI_API_KEY=your-actual-api-key-here
   OPENAI_MODEL=gpt-3.5-turbo
   ```

   **OR** for Google Gemini:
   ```env
   LLM_PROVIDER=google
   GOOGLE_API_KEY=your-google-api-key-here
   GOOGLE_MODEL=gemini-pro
   ```

**Getting API Keys:**
- OpenAI: [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- Google AI: [makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)

## ⚡ Quick Start

### Running the Application

```bash
python app.py
```

The application will start and display:
```
Running on local URL:  http://127.0.0.1:7860
```

Open your browser and navigate to `http://127.0.0.1:7860`

### Basic Workflow

1. **Upload Documents**
   - Click on "Upload Documents" tab
   - Select one or more files (PDF, TXT, or DOCX)
   - Click "Process Documents"
   - Wait for confirmation message

2. **Ask Questions**
   - Switch to "Chat" tab
   - Type your question in the text box
   - Press Enter or click "Send"
   - View the answer with source citations

3. **Manage Database**
   - Go to "Database" tab
   - View statistics about stored documents
   - Clear database if needed

## ⚙️ Configuration

All configuration is managed in `src/config.py`. Key settings:

### Text Chunking
```python
CHUNK_SIZE_MIN = 500      # Minimum characters per chunk
CHUNK_SIZE_MAX = 800      # Maximum characters per chunk
CHUNK_OVERLAP_PERCENT = 10  # 10% overlap between chunks
```

### Embedding Model
```python
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
```

### Retrieval
```python
TOP_K_RETRIEVAL = 5  # Number of chunks to retrieve (3-5 as per spec)
```

### LLM Settings
```python
MAX_TOKENS = 1000
TEMPERATURE = 0.7
```

### Gradio UI
```python
GRADIO_SHARE = False  # Set True for public URL
GRADIO_SERVER_PORT = 7860
```

## 📖 Usage

### Using as a Library

```python
from src import DocumentIngestor, TextChunker, VectorStore, RAGEngine

# 1. Ingest a document
ingestor = DocumentIngestor()
text, metadata = ingestor.ingest_document("document.pdf")

# 2. Chunk the text
chunker = TextChunker()
chunks = chunker.chunk_text(text, metadata["source"], metadata)

# 3. Store in vector database
vector_store = VectorStore()
vector_store.add_chunks(chunks)

# 4. Create RAG engine and ask questions
rag_engine = RAGEngine(vector_store)
result = rag_engine.generate_answer("What is the main topic?")
print(result["answer"])
print("Sources:", result["sources"])
```

### Advanced Examples

#### Processing Multiple Documents
```python
from src import ingest_documents, chunk_documents, VectorStore

# Ingest multiple files
file_paths = ["doc1.pdf", "doc2.txt", "doc3.docx"]
documents = ingest_documents(file_paths)

# Chunk all documents
all_chunks = chunk_documents(documents)

# Store in vector database
vector_store = VectorStore()
vector_store.add_chunks(all_chunks)
```

#### Custom Chunk Sizes
```python
from src import TextChunker

# Create custom chunker
chunker = TextChunker(
    chunk_size_min=300,
    chunk_size_max=600,
    overlap_percent=15
)

chunks = chunker.chunk_text(text, source="custom.txt")
```

#### Filtering Search by Source
```python
# Search only in specific document
results = vector_store.search(
    query="machine learning",
    top_k=3,
    filter_metadata={"source": "ml_paper.pdf"}
)
```

## 📁 Project Structure

```
capstone project 1/
├── app.py                      # Main Gradio application
├── requirements.txt            # Python dependencies
├── .env.example               # Environment variables template
├── .gitignore                 # Git ignore rules
├── README.md                  # This file
├── DEPLOYMENT.md              # Deployment guide
│
├── src/                       # Source code
│   ├── __init__.py           # Package initialization
│   ├── config.py             # Configuration settings
│   ├── document_ingestion.py # Document loading (PDF/TXT/DOCX)
│   ├── text_chunking.py      # Text chunking logic
│   ├── vector_store.py       # ChromaDB vector storage
│   ├── rag_engine.py         # RAG retrieval & LLM integration
│   └── utils.py              # Utility functions
│
├── tests/                     # Test suite
│   ├── conftest.py           # Pytest configuration
│   ├── test_document_ingestion.py
│   ├── test_text_chunking.py
│   ├── test_vector_store.py
│   ├── test_rag_engine.py
│   └── test_integration.py   # End-to-end tests
│
├── data/                      # Sample documents (user uploads)
└── chroma_db/                # Vector database storage
```

## 🧪 Testing

The project includes comprehensive test suites for all components, including both basic unit tests and enhanced tests with mocking capabilities.

### Quick Start - Run All Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with detailed output showing test progress
pytest -v --tb=short
```

### Test Files Overview

| Test File | Purpose | Test Count | Type |
|-----------|---------|------------|------|
| `test_retriever.py` | Basic retriever unit tests | 13 | Unit |
| `test_retriever_enhanced.py` | Enhanced retriever tests with mocks | 10 | Mock/Unit |
| `test_generator.py` | Basic generator unit tests | 21 | Unit |
| `test_generator_enhanced.py` | Enhanced generator tests with synthetic data | 20 | Mock/Unit |
| **Total** | **All test suites** | **64** | **Combined** |

### Running Specific Test Suites

#### Retriever Tests

```bash
# Basic retriever tests (with real ChromaDB)
pytest test_retriever.py -v

# Enhanced retriever tests (with mocked ChromaDB)
pytest test_retriever_enhanced.py -v

# Both retriever test suites
pytest test_retriever.py test_retriever_enhanced.py -v
```

**What's Tested:**
- ✅ `get_top_k()` function returns expected chunks for known queries
- ✅ Edge case: empty collection (no documents indexed)
- ✅ Semantic search with similarity scoring
- ✅ Keyword fallback when similarity is low
- ✅ Parameter validation (k value, empty queries, etc.)
- ✅ Return format consistency

#### Generator Tests

```bash
# Basic generator tests
pytest test_generator.py -v

# Enhanced generator tests with synthetic data
pytest test_generator_enhanced.py -v

# Both generator test suites
pytest test_generator.py test_generator_enhanced.py -v
```

**What's Tested:**
- ✅ Answer generation with synthetic retrieved chunks
- ✅ Answer contains expected keywords from source material
- ✅ Sources list is non-empty and properly formatted
- ✅ Confidence scores are reasonable (0-1 range)
- ✅ Rule-based fallback when LLM unavailable
- ✅ Edge cases (empty chunks, special characters, very long text)
- ✅ Prompt template formatting

### Test Categories

#### 1. Mock Tests (No External Dependencies)

These tests use `unittest.mock` to simulate ChromaDB, SentenceTransformer, and other components without requiring actual databases or models:

```bash
# Retriever mocking tests
pytest test_retriever_enhanced.py::TestRetrieverWithMocking -v

# Generator synthetic data tests
pytest test_generator_enhanced.py::TestGenerateAnswerWithSyntheticData -v
```

**Benefits:**
- ⚡ Fast execution (no model loading or DB queries)
- 🔒 Deterministic results (controlled test data)
- 🧪 Isolated component testing
- ✅ No API keys or setup required

**Example Mocked Test:**
```python
def test_get_top_k_returns_expected_chunks(self, mock_retriever):
    """Test that get_top_k returns expected chunks for known query."""
    query = "What is machine learning?"
    results = mock_retriever.get_top_k(query, k=2)
    
    # Verify we got results
    assert len(results) == 2
    
    # Verify first result contains expected content
    assert 'Machine learning' in results[0]['text']
    assert results[0]['metadata']['source'] == 'ml_basics.txt'
```

#### 2. Integration Tests (Real Components)

These tests use actual ChromaDB and embeddings (slower but validate real behavior):

```bash
# Integration tests with real database
pytest test_retriever.py -v

# Integration tests with real generator
pytest test_generator.py -v
```

**Requirements:**
- ChromaDB directory (`./chroma_db`)
- Indexed documents (run `ingestion.py` first)
- Sentence-transformers model downloaded

### Running Tests with Coverage

```bash
# Generate coverage report
pytest --cov=. --cov-report=html

# Open coverage report in browser
start htmlcov/index.html   # Windows
open htmlcov/index.html    # macOS
xdg-open htmlcov/index.html  # Linux
```

**Current Coverage:**
- ✅ `ingestion.py`: Tested (document loading, chunking)
- ✅ `embeddings_and_chroma_setup.py`: Tested (embedding computation, storage)
- ✅ `retriever.py`: 100% coverage (13 + 10 tests)
- ✅ `generator.py`: 100% coverage (21 + 20 tests)
- ✅ `rag_app.py`: Integration tested

### Test Fixtures

The test suites use pytest fixtures for reusable test data:

```python
# Synthetic ML chunks
@pytest.fixture
def ml_chunks(self):
    return [
        {
            "id": "ml_doc_0",
            "text": "Machine learning is a method of data analysis...",
            "metadata": {"source": "ml_intro.pdf", "chunk_index": 0},
            "score": 0.92
        },
        # ... more chunks
    ]

# Mocked ChromaDB collection
@pytest.fixture
def mock_chroma_collection(self):
    mock_collection = Mock()
    mock_collection.query.return_value = {
        'ids': [['doc1_0', 'doc2_0']],
        'documents': [['Machine learning...', 'Artificial intelligence...']],
        # ... mocked return data
    }
    return mock_collection
```

### Debugging Failed Tests

```bash
# Show full traceback
pytest -v --tb=long

# Stop at first failure
pytest -x

# Run only failed tests from last run
pytest --lf

# Run specific test by name
pytest test_retriever_enhanced.py::TestRetrieverWithMocking::test_get_top_k_returns_expected_chunks -v
```

### Adding New Tests

When adding new functionality, create corresponding tests:

```python
# test_new_feature.py
import pytest
from my_module import my_function

def test_my_function():
    """Test that my_function works correctly."""
    result = my_function("input")
    assert result == "expected_output"
```

Then run:
```bash
pytest test_new_feature.py -v
```

### Continuous Integration

For CI/CD pipelines, use:

```bash
# Run all tests with coverage and strict warnings
pytest -v --cov=. --cov-report=xml --cov-report=term --strict-warnings

# Exit with error if coverage below threshold
pytest --cov=. --cov-fail-under=80
```

### Test Summary

**Total Tests: 64**
- ✅ **10** enhanced retriever tests with mocking (test_retriever_enhanced.py)
- ✅ **13** basic retriever tests with real DB (test_retriever.py)
- ✅ **20** enhanced generator tests with synthetic data (test_generator_enhanced.py)
- ✅ **21** basic generator tests with real components (test_generator.py)

**All tests passing** ✨

Example output:
```
================================================== test session starts ===================================================
collected 64 items

test_retriever.py::TestRetriever::test_init PASSED                                                                 [  1%]
test_retriever.py::TestRetriever::test_get_top_k_format PASSED                                                     [  3%]
...
test_generator_enhanced.py::TestPromptTemplates::test_simple_prompt_formatting PASSED                             [100%]

================================================== 64 passed in 59.11s ===================================================
```

---

## 📚 API Reference

### DocumentIngestor

```python
ingestor = DocumentIngestor()

# Ingest a single document
text, metadata = ingestor.ingest_document("file.pdf")

# Validate file before processing
is_valid = ingestor.validate_file("file.txt")
```

### TextChunker

```python
chunker = TextChunker(
    chunk_size_min=500,
    chunk_size_max=800,
    overlap_percent=10
)

# Chunk text
chunks = chunker.chunk_text(
    text="Your document text...",
    source="document.pdf",
    metadata={"author": "John Doe"}
)
```

### VectorStore

```python
vector_store = VectorStore(
    collection_name="my_docs",
    persist_directory="./my_db"
)

# Add chunks
vector_store.add_chunks(chunks)

# Search
results = vector_store.search("query", top_k=5)

# Manage database
stats = vector_store.get_stats()
sources = vector_store.get_all_sources()
vector_store.delete_by_source("old_doc.pdf")
vector_store.clear()
```

### RAGEngine

```python
rag_engine = RAGEngine(
    vector_store=vector_store,
    llm_provider="openai",
    top_k=5
)

# Generate answer
result = rag_engine.generate_answer(
    query="What is machine learning?",
    include_sources=True
)

print(result["answer"])
print(result["sources"])
print(result["context_found"])
```

## 🌐 Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions including:
- Docker containerization
- Cloud deployment (AWS, GCP, Azure)
- Production configurations
- Scaling strategies

## 🔧 Troubleshooting

### Common Issues

**1. Import Errors**
```
ModuleNotFoundError: No module named 'chromadb'
```
**Solution**: Install dependencies: `pip install -r requirements.txt`

**2. API Key Errors**
```
ValueError: OPENAI_API_KEY not found
```
**Solution**: Create `.env` file with your API key

**3. Out of Memory**
```
ChromaDB embedding error: Out of memory
```
**Solution**: Process documents in smaller batches or increase system RAM

**4. Port Already in Use**
```
OSError: [Errno 48] Address already in use
```
**Solution**: Change port in `.env`: `GRADIO_SERVER_PORT=7861`

**5. PDF Extraction Issues**
```
No text could be extracted from PDF
```
**Solution**: PDF may be image-based. Use OCR tools or convert to text first.

### Getting Help

- Check logs in the console output
- Review error messages in the UI
- Ensure all dependencies are installed correctly
- Verify API keys are valid and have credits

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **Sentence Transformers** for embeddings
- **ChromaDB** for vector storage
- **Gradio** for the UI framework
- **OpenAI / Google** for LLM APIs
- **PyPDF2** and **python-docx** for document processing

## 🌐 Deployment

### Hugging Face Spaces (Recommended - FREE)

This app is ready to deploy on Hugging Face Spaces:

1. Create account at https://huggingface.co/join
2. Create new Space with Gradio SDK
3. Push your code:
```bash
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/rag-chatbot
git push hf main
```

**See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for detailed deployment instructions for:**
- Hugging Face Spaces (FREE)
- Render (FREE tier)
- Railway (FREE tier)
- Heroku (Paid)

## 📊 Document Collection

The system includes **20 comprehensive documents** (139 indexed chunks):

**Technical Documents (5):**
- Machine Learning, Python, AI, Deep Learning, Data Science

**Company Documents (15):**
- HR Policies, Financial Reports, Salary Data, Performance Metrics, Training Programs, Strategic Initiatives, and more

## 📧 Contact

**Author**: Jagjeet Jena  
**GitHub**: [@jagjeetjenagit](https://github.com/jagjeetjenagit)  
**Repository**: [rag-chatbot-capstone](https://github.com/jagjeetjenagit/rag-chatbot-capstone)

## 📞 Support

For issues, questions, or suggestions:
- Open an issue in the repository
- Check the documentation
- Review existing issues for solutions

---

**⭐ Star this repo if you find it useful!**

**🚀 Deploy now:** [Deployment Guide](DEPLOYMENT_GUIDE.md)

**Built with ❤️ for the RAG Chatbot Capstone Project**

