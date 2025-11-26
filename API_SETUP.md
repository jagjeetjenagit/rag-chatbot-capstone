# 🔑 API Setup Guide

This guide will help you configure API keys for the RAG Chatbot to use AI language models.

## 📋 Table of Contents

- [Overview](#overview)
- [Option 1: OpenAI (Recommended)](#option-1-openai-recommended)
- [Option 2: Google Gemini](#option-2-google-gemini)
- [Option 3: Ollama (Free Local LLM)](#option-3-ollama-free-local-llm)
- [Option 4: Rule-Based (No API Required)](#option-4-rule-based-no-api-required)
- [Configuration](#configuration)
- [Testing](#testing)

---

## Overview

The RAG Chatbot can work with multiple LLM providers:

| Provider | Cost | Quality | Speed | Setup Difficulty |
|----------|------|---------|-------|------------------|
| **OpenAI** | 💰 Paid | ⭐⭐⭐⭐⭐ | ⚡⚡⚡ | Easy |
| **Google Gemini** | 💰 Paid/Free tier | ⭐⭐⭐⭐ | ⚡⚡⚡ | Easy |
| **Ollama** | 🆓 Free | ⭐⭐⭐ | ⚡⚡ | Medium |
| **Rule-Based** | 🆓 Free | ⭐⭐ | ⚡⚡⚡⚡ | None |

---

## Option 1: OpenAI (Recommended)

### Step 1: Get API Key

1. Go to [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Sign up or log in
3. Click **"Create new secret key"**
4. Copy the key (starts with `sk-...`)

### Step 2: Add to config.py

Open [`config.py`](config.py) and add your key:

```python
# OpenAI API Configuration
OPENAI_API_KEY = "sk-your-actual-key-here"  # ← Paste your key here
OPENAI_MODEL = "gpt-3.5-turbo"  # or "gpt-4" for better quality
```

### Step 3: Set Provider

```python
LLM_PROVIDER = "openai"  # ← Use OpenAI
```

### Pricing

- **GPT-3.5-Turbo**: ~$0.002 per 1K tokens (~$0.01 per 10 questions)
- **GPT-4**: ~$0.03 per 1K tokens (~$0.15 per 10 questions)

💡 **Tip**: Start with gpt-3.5-turbo for testing (cheaper and faster)

---

## Option 2: Google Gemini

### Step 1: Get API Key

1. Go to [https://makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)
2. Sign in with Google account
3. Click **"Create API key"**
4. Copy the key

### Step 2: Add to config.py

```python
# Google Gemini API Configuration
GOOGLE_API_KEY = "your-google-api-key-here"  # ← Paste your key here
GOOGLE_MODEL = "gemini-pro"
```

### Step 3: Set Provider

```python
LLM_PROVIDER = "google"  # ← Use Google Gemini
```

### Step 4: Install Package

```bash
pip install google-generativeai
```

### Pricing

- **Gemini Pro**: Free tier available (60 requests/minute)
- Paid: ~$0.001 per 1K tokens

---

## Option 3: Ollama (Free Local LLM)

Run AI models locally on your computer - **100% free, no internet required!**

### Step 1: Install Ollama

**Windows:**
1. Download from [https://ollama.ai/download/windows](https://ollama.ai/download/windows)
2. Run installer
3. Ollama will start automatically

**Mac:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### Step 2: Download a Model

```bash
# Recommended: Llama 2 (7GB)
ollama pull llama2

# Or try Mistral (smaller, faster)
ollama pull mistral

# Or CodeLlama for coding questions
ollama pull codellama
```

### Step 3: Configure

In [`config.py`](config.py):

```python
LLM_PROVIDER = "ollama"  # ← Use Ollama
OLLAMA_MODEL = "llama2"  # or "mistral", "codellama"
```

### Step 4: Verify Running

```bash
# Check if Ollama is running
ollama list

# Test it
ollama run llama2
```

### Pros & Cons

✅ **Pros:**
- Completely free
- Works offline
- Privacy (data never leaves your computer)

❌ **Cons:**
- Requires powerful computer (8GB+ RAM)
- Slower than cloud APIs
- Lower quality than GPT-4

---

## Option 4: Rule-Based (No API Required)

The chatbot includes a **built-in rule-based answer generator** that works without any API!

### How It Works

- Extracts keywords from your question
- Finds relevant sentences in retrieved chunks
- Combines them into an answer
- Fast and free, but less sophisticated

### Configure

In [`config.py`](config.py):

```python
LLM_PROVIDER = "fallback"  # ← Use rule-based only
```

Or just leave API keys empty - it will auto-fallback!

---

## Configuration

### Option 1: Direct Edit (Simple)

1. Open [`config.py`](config.py)
2. Add your API key
3. Save the file

### Option 2: Environment Variables (Secure)

For production or sharing code:

**Windows PowerShell:**
```powershell
$env:OPENAI_API_KEY = "sk-your-key-here"
$env:GOOGLE_API_KEY = "your-google-key"
```

**Mac/Linux:**
```bash
export OPENAI_API_KEY="sk-your-key-here"
export GOOGLE_API_KEY="your-google-key"
```

The app will automatically read these!

### Auto Provider Selection

Set to `"auto"` to try providers in order:

```python
LLM_PROVIDER = "auto"  # Tries: OpenAI → Google → Ollama → Fallback
```

---

## Testing

### Test Configuration

```bash
python config.py
```

Output shows what's configured:
```
================================================================================
RAG CHATBOT CONFIGURATION
================================================================================

📁 Documents Directory: data/documents
🔧 LLM Provider: openai
🗄️  ChromaDB Path: ./chroma_db
🔍 Embedding Model: all-MiniLM-L6-v2

================================================================================
API STATUS
================================================================================
OpenAI: ✅ Configured
Google: ❌ Not configured
Ollama: ✅ Available (local)
```

### Test APIs

```bash
python llm_api.py
```

This sends a test question to configured providers:
```
================================================================================
TESTING LLM API INTEGRATIONS
================================================================================

🧪 Testing OpenAI...
✅ OpenAI Response: Machine learning is a type of artificial intelligence...

🧪 Testing Google Gemini...
⏭️  Skipping Google Gemini (not configured)

🧪 Testing Ollama (local)...
❌ Ollama failed (is Ollama running?)
```

---

## Quick Start Examples

### Scenario 1: Student (Free)

```python
# config.py
LLM_PROVIDER = "fallback"  # No API needed
```

**Cost**: $0  
**Setup**: 0 minutes

### Scenario 2: Hobbyist (Free Local)

```python
# config.py
LLM_PROVIDER = "ollama"
OLLAMA_MODEL = "llama2"
```

**Cost**: $0  
**Setup**: 10 minutes (download model)

### Scenario 3: Professional (Best Quality)

```python
# config.py
LLM_PROVIDER = "openai"
OPENAI_API_KEY = "sk-your-key"
OPENAI_MODEL = "gpt-4"
```

**Cost**: ~$0.15 per 10 questions  
**Setup**: 2 minutes

---

## Troubleshooting

### "OpenAI API key not configured"

- Check you added the key to [`config.py`](config.py)
- Key should start with `sk-`
- No extra quotes or spaces

### "Ollama connection failed"

```bash
# Check if running
ollama list

# Start Ollama
ollama serve

# In new terminal
ollama pull llama2
```

### "All LLM providers failed"

The app will automatically use **rule-based fallback** - answers will still work, just less sophisticated!

---

## Security Best Practices

1. **Never commit API keys to Git**
   - Keys are in [`config.py`](config.py) which is in `.gitignore`
   
2. **Use environment variables in production**
   - More secure than hardcoding
   
3. **Rotate keys regularly**
   - Regenerate keys every few months
   
4. **Monitor usage**
   - Check OpenAI/Google dashboards for unexpected usage

---

## Need Help?

- **OpenAI Issues**: [https://help.openai.com](https://help.openai.com)
- **Google Issues**: [https://ai.google.dev](https://ai.google.dev)
- **Ollama Issues**: [https://github.com/ollama/ollama](https://github.com/ollama/ollama)

---

**🎉 Ready to use your RAG Chatbot with AI! 🎉**

Choose your option above, configure, and run:
```bash
python rag_app.py
```
