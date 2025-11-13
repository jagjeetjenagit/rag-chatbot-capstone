# RAG Chatbot - Quick Reference Card

## 🚀 Installation & Setup

### Windows PowerShell
```powershell
# Quick setup
.\setup.ps1

# Manual setup
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env with your API key
```

### Linux/Mac
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API key
```

## ⚙️ Configuration (.env)

```env
# Required
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here

# Optional
OPENAI_MODEL=gpt-3.5-turbo
GRADIO_SERVER_PORT=7860
```

## 🎯 Running the App

```bash
# Start web UI
python app.py

# Run example script
python example_usage.py

# Run tests
pytest tests/ -v
```

## 📝 Key Features

| Feature | Implementation |
|---------|----------------|
| **Chunk Size** | 500-800 characters |
| **Overlap** | 10% between chunks |
| **Embedding Model** | all-MiniLM-L6-v2 |
| **Vector DB** | ChromaDB |
| **Top-K Retrieval** | 3-5 chunks |
| **Supported Formats** | PDF, TXT, DOCX |
| **LLM Providers** | OpenAI, Google |

## 💻 Code Snippets

### Basic Usage
```python
from src import DocumentIngestor, TextChunker, VectorStore, RAGEngine

# Ingest document
ingestor = DocumentIngestor()
text, metadata = ingestor.ingest_document("doc.pdf")

# Chunk text
chunker = TextChunker()
chunks = chunker.chunk_text(text, metadata["source"])

# Store in vector DB
vector_store = VectorStore()
vector_store.add_chunks(chunks)

# Ask questions
rag = RAGEngine(vector_store)
result = rag.generate_answer("What is this about?")
print(result["answer"])
```

### Search with Filter
```python
# Search in specific document
results = vector_store.search(
    "machine learning",
    filter_metadata={"source": "doc.pdf"}
)
```

### Database Management
```python
# Get stats
stats = vector_store.get_stats()

# Get all sources
sources = vector_store.get_all_sources()

# Delete document
vector_store.delete_by_source("old.pdf")

# Clear all
vector_store.clear()
```

## 🧪 Testing Commands

```bash
# All tests
pytest tests/ -v

# Specific module
pytest tests/test_document_ingestion.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Integration only
pytest tests/test_integration.py -v
```

## 📁 Project Structure

```
src/
  config.py           - Configuration settings
  document_ingestion.py - PDF/TXT/DOCX processing
  text_chunking.py    - Text chunking with overlap
  vector_store.py     - ChromaDB + embeddings
  rag_engine.py       - Retrieval + LLM
  utils.py            - Helper functions

tests/
  test_*.py           - Unit tests
  test_integration.py - E2E tests
  conftest.py         - Test config

app.py                - Gradio web UI
example_usage.py      - Code examples
```

## 🔧 Common Tasks

### Add New Document
1. Upload via UI, OR
2. Place in `data/` folder
3. Process programmatically

### Change Chunk Size
Edit `src/config.py`:
```python
CHUNK_SIZE_MIN = 300
CHUNK_SIZE_MAX = 600
```

### Switch LLM Provider
Edit `.env`:
```env
LLM_PROVIDER=google
GOOGLE_API_KEY=your-key
```

### Change Retrieval Count
Edit `src/config.py`:
```python
TOP_K_RETRIEVAL = 3  # 3-5 recommended
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Import errors | `pip install -r requirements.txt` |
| API key error | Check `.env` file |
| Port in use | Change `GRADIO_SERVER_PORT` |
| No PDF text | PDF may need OCR |
| Memory error | Reduce batch size |

## 🌐 URLs

| Service | URL |
|---------|-----|
| Local UI | http://127.0.0.1:7860 |
| OpenAI Keys | https://platform.openai.com/api-keys |
| Google AI Keys | https://makersuite.google.com/app/apikey |

## 📚 Documentation

- **README.md** - Full documentation
- **DEPLOYMENT.md** - Deployment guides
- **PROJECT_SUMMARY.md** - Project overview
- **Code comments** - Inline docs

## 🚀 Deployment

### Docker
```bash
docker build -t rag-chatbot .
docker run -p 7860:7860 --env-file .env rag-chatbot
```

### Production (systemd)
```bash
sudo cp rag-chatbot.service /etc/systemd/system/
sudo systemctl enable rag-chatbot
sudo systemctl start rag-chatbot
```

## ⚡ Performance Tips

1. **Faster responses**: Reduce `TOP_K_RETRIEVAL` to 3
2. **More context**: Increase `TOP_K_RETRIEVAL` to 5
3. **Better chunks**: Adjust chunk sizes
4. **Cost savings**: Use `gpt-3.5-turbo` vs `gpt-4`

## 🔒 Security Checklist

- [ ] API keys in `.env` (not in code)
- [ ] `.env` in `.gitignore`
- [ ] File size limits enabled
- [ ] File type validation active
- [ ] HTTPS in production
- [ ] Regular key rotation

## 📊 Monitoring

Check these regularly:
- Application logs
- Disk space (`chroma_db/` folder)
- API usage/costs
- Error rates
- Response times

## 🎯 Best Practices

1. ✅ Always activate virtual environment
2. ✅ Keep dependencies updated
3. ✅ Test after making changes
4. ✅ Backup `chroma_db/` folder
5. ✅ Monitor API costs
6. ✅ Clear old documents periodically

## 💡 Tips

- Start with sample document in `data/`
- Test with `example_usage.py` first
- Use database management tab to view stats
- Check logs if something fails
- Small documents = faster testing

---

**Need Help?** Check README.md or DEPLOYMENT.md for detailed information.
