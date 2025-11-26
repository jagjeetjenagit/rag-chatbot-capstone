# 🎯 API Configuration - Simple Guide

## ✅ You're All Set Up!

Your RAG Chatbot is ready to use. Here's what you need to know:

## 📝 **Where to Add API Keys**

### **Only ONE file to edit:** `.env`

That's it! Just open `.env` and add your API key.

## 🔑 How to Add an API Key

### Step 1: Choose a Provider

Pick **ONE** of these options:

| Provider | Cost | Quality | Link |
|----------|------|---------|------|
| **OpenAI** | $0.002 / 1K tokens | ⭐⭐⭐⭐⭐ Best | https://platform.openai.com/api-keys |
| **Google Gemini** | Free tier available | ⭐⭐⭐⭐ Good | https://makersuite.google.com/app/apikey |
| **Ollama** | 100% Free (local) | ⭐⭐⭐ OK | https://ollama.ai |
| **No API** | Free | ⭐⭐ Basic | Just run the app! |

### Step 2: Get Your API Key

**For OpenAI:**
1. Go to https://platform.openai.com/api-keys
2. Sign up / Log in
3. Click "Create new secret key"
4. Copy the key (starts with `sk-`)

**For Google Gemini:**
1. Go to https://makersuite.google.com/app/apikey
2. Sign in with Google
3. Click "Create API Key"
4. Copy the key

**For Ollama (Local):**
1. Download from https://ollama.ai
2. Install Ollama
3. Run: `ollama pull llama2`
4. No API key needed!

### Step 3: Add to .env File

Open `.env` file and paste your key:

```env
# For OpenAI:
OPENAI_API_KEY=sk-your-actual-key-here

# For Google:
GOOGLE_API_KEY=your-actual-key-here

# For Ollama (no key needed):
OLLAMA_BASE_URL=http://localhost:11434
```

**Save the file!**

### Step 4: Run the App

```bash
# Activate virtual environment first:
.venv\Scripts\Activate.ps1

# Then run:
python start.py
```

Open http://localhost:7860 🎉

## ❓ FAQ

### Q: Which file should I edit for API keys?
**A:** Only `.env` - nowhere else!

### Q: Can I use the app without an API key?
**A:** Yes! It will use rule-based responses (basic quality).

### Q: How do I know if my API key is working?
**A:** Run `python config.py` to check status.

### Q: Is my API key safe?
**A:** Yes! The `.env` file is in `.gitignore` and won't be committed to Git.

### Q: Can I switch between providers?
**A:** Yes! Just change `LLM_PROVIDER` in `.env`:
```env
LLM_PROVIDER=openai    # Use OpenAI
LLM_PROVIDER=google    # Use Google
LLM_PROVIDER=ollama    # Use Ollama
LLM_PROVIDER=auto      # Try all (default)
```

## 🔍 Quick Test

After adding your API key:

```bash
# Test configuration
python config.py

# Test API connection
python llm_api.py
```

You should see ✅ for your configured provider!

## 📁 File Reference

- **`.env`** ← ADD YOUR API KEYS HERE (only file to edit!)
- `.env.example` - Template (don't edit)
- `config.py` - Auto-loads from .env (don't edit)
- `start.py` - Launch script
- `SETUP.md` - Detailed setup guide
- `API_SETUP.md` - Complete API documentation

## 🚀 Ready to Go!

1. Add your API key to `.env`
2. Run `python start.py`
3. Open http://localhost:7860
4. Start chatting! 💬

---

**Still confused?** See `SETUP.md` for step-by-step instructions.
