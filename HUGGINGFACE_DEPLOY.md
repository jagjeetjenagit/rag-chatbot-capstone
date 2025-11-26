# 🚀 Quick Deploy to Hugging Face Spaces

## Step 1: Create Hugging Face Account

1. Go to https://huggingface.co/join
2. Sign up with email or GitHub
3. Verify your email

## Step 2: Create a New Space

1. Go to https://huggingface.co/spaces
2. Click **"Create new Space"**
3. Fill in the details:
   - **Owner:** Your username
   - **Space name:** `rag-chatbot` (or any name you prefer)
   - **License:** Apache 2.0 (or your choice)
   - **Select SDK:** **Gradio** ⚠️ IMPORTANT!
   - **Space hardware:** CPU basic (Free)
   - **Visibility:** Public (or Private if you prefer)
4. Click **"Create Space"**

## Step 3: Prepare Your Repository

Your repository is already set up! But let's create one more file for Hugging Face:

Create `app.py` (Hugging Face looks for this by default):

```python
# This file redirects to our main application
from app_github import *

if __name__ == "__main__":
    main()
```

## Step 4: Push to Hugging Face

Open PowerShell in your project directory and run:

```powershell
# Add Hugging Face remote (replace YOUR_USERNAME with your HF username)
git remote add huggingface https://huggingface.co/spaces/YOUR_USERNAME/rag-chatbot

# Push to Hugging Face
git push huggingface main
```

**Example:**
```powershell
git remote add huggingface https://huggingface.co/spaces/jagjeetjena/rag-chatbot
git push huggingface main
```

## Step 5: Wait for Build

1. Go to your Space URL: `https://huggingface.co/spaces/YOUR_USERNAME/rag-chatbot`
2. You'll see "Building..." status
3. First build takes **5-10 minutes**
4. Watch the build logs in the "Logs" tab

## Step 6: Access Your Deployed App

Once build completes:
- **Your App URL:** `https://huggingface.co/spaces/YOUR_USERNAME/rag-chatbot`
- The Gradio interface will load automatically
- ChromaDB will index documents on first run (~30 seconds)

## 🎉 That's It!

Your RAG chatbot is now live and publicly accessible!

---

## 🔧 Troubleshooting

### Build Fails

**Error: "No app.py found"**
- Create `app.py` as shown in Step 3
- Or rename `app_github.py` to `app.py`

**Error: "Requirements not found"**
- Ensure `requirements.txt` is in root directory
- Check file is committed to git

**Error: "Python version mismatch"**
- Hugging Face will use Python 3.10 by default
- Your `runtime.txt` specifies Python 3.11.7
- If issues occur, update to `python-3.10.12`

### App Runs But No Documents

**Symptom: "No relevant documents found"**
- Check `data/documents/` folder exists
- Verify `.gitignore` allows `.txt` files
- Check build logs for indexing errors

### Out of Memory

**Symptom: App crashes or restarts**
- Free tier has 16GB RAM limit (usually enough)
- If needed, reduce batch size in code
- Or upgrade to better hardware (paid)

---

## 📝 Update Your Deployed App

To update after making changes:

```powershell
# Make your changes locally
# Commit changes
git add .
git commit -m "Your update message"

# Push to GitHub
git push origin main

# Push to Hugging Face
git push huggingface main
```

The Space will automatically rebuild with your changes.

---

## 🎨 Customize Your Space

### Add a README

Hugging Face displays `README.md` on your Space page. Add custom content:

```markdown
---
title: RAG Chatbot
emoji: 📚
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 3.50.2
app_file: app.py
pinned: false
---

# My RAG Chatbot

Ask questions about our document collection!
```

### Add a Card Image

1. Create a preview image (1200x630px recommended)
2. Save as `thumbnail.png` in your repo
3. Commit and push

---

## 🌟 Best Practices

✅ **Test locally first** - Ensure app works before deploying  
✅ **Keep dependencies minimal** - Faster builds  
✅ **Monitor logs** - Watch for errors or warnings  
✅ **Use versioned requirements** - Avoid breaking changes  
✅ **Add examples** - Help users understand your app  
✅ **Document your Space** - Good README = more users  

---

## 🔗 Useful Links

- **Your GitHub Repo:** https://github.com/jagjeetjenagit/rag-chatbot-capstone
- **Hugging Face Docs:** https://huggingface.co/docs/hub/spaces
- **Gradio Docs:** https://gradio.app/docs
- **HF Spaces Examples:** https://huggingface.co/spaces

---

## 💡 Alternative: Use the Hugging Face CLI

```powershell
# Install HF CLI
pip install huggingface_hub

# Login
huggingface-cli login

# Upload your space
huggingface-cli upload-space . YOUR_USERNAME/rag-chatbot
```

---

**Ready? Go create your Space now! 🚀**

**Your repo is at:** https://github.com/jagjeetjenagit/rag-chatbot-capstone
