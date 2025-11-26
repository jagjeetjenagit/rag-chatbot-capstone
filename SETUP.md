# RAG Chatbot - Quick Setup Guide

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## 🚀 Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

**Option A: Using .env file (Recommended)**

1. Copy the example file:
   ```bash
   # On Windows PowerShell:
   Copy-Item .env.example .env
   
   # On Linux/Mac:
   cp .env.example .env
   ```

2. Edit `.env` file and add your API key:
   ```env
   # Choose ONE option:
   
   # Option 1: OpenAI (Best quality, ~$0.002 per 1K tokens)
   OPENAI_API_KEY=sk-your-key-here
   
   # Option 2: Google Gemini (Free tier available)
   GOOGLE_API_KEY=your-key-here
   
   # Option 3: Ollama (100% free, runs locally)
   # Just install Ollama from https://ollama.ai
   # No API key needed!
   ```

**Option B: Without API keys**

The app will work without any API keys using a rule-based fallback system. Just skip the .env configuration!

## 🎯 Running the App

### Simple Start

```bash
python start.py
```

The script will:
- ✅ Validate your configuration
- ✅ Show which APIs are configured
- ✅ Launch the chatbot on http://localhost:7860

### Manual Start

```bash
python rag_app.py
```

## 🔑 Getting API Keys

### OpenAI (Recommended)
1. Go to https://platform.openai.com/api-keys
2. Sign up or log in
3. Click "Create new secret key"
4. Copy the key and add to `.env` file
5. **Pricing**: ~$0.002 per 1K tokens (very affordable)

### Google Gemini (Free Option)
1. Go to https://makersuite.google.com/app/apikey
2. Sign in with Google account
3. Click "Create API Key"
4. Copy the key and add to `.env` file
5. **Free tier**: 60 requests per minute

### Ollama (Local & Free)
1. Download from https://ollama.ai
2. Install Ollama
3. Run: `ollama pull llama2`
4. No API key needed - runs on your computer!

## 📁 Project Structure

```
capstone-project-1/
├── .env                    # Your API keys (DO NOT commit!)
├── .env.example           # Template for .env
├── config.py              # Loads configuration from .env
├── llm_api.py            # Unified LLM interface
├── start.py              # Quick start script
├── rag_app.py            # Main application
├── SETUP.md              # This file
└── API_SETUP.md          # Detailed API guide
```

## 🧪 Testing Configuration

Test if your API keys work:

```bash
# Check configuration
python config.py

# Test API connections
python llm_api.py
```

## ❓ Troubleshooting

### "Module not found" error
```bash
pip install -r requirements.txt
```

### API key not working
1. Make sure `.env` file is in the project root
2. Check that `python-dotenv` is installed: `pip install python-dotenv`
3. Verify API key has no extra spaces or quotes
4. Run `python config.py` to see what's loaded

### Port 7860 already in use
Edit `.env` and add:
```env
GRADIO_PORT=7861
```

### Slow first startup
- The first run downloads the embedding model (~100MB)
- This is normal and only happens once
- Takes 5-10 seconds

## 📚 Need More Help?

- **API Setup Details**: See `API_SETUP.md`
- **General Issues**: Check the logs in the terminal
- **Test Suite**: Run `pytest` to verify everything works

## 🎉 You're Ready!

Run `python start.py` and open http://localhost:7860 in your browser!
