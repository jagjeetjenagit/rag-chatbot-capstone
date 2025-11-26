# Deployment Guide - RAG Chatbot

This guide covers deploying your RAG chatbot to various hosting platforms.

## 🚀 Deployment Options

### 1. Hugging Face Spaces (Recommended - FREE)

Hugging Face Spaces is the easiest and free option for hosting Gradio apps.

**Steps:**

1. **Create a Hugging Face account** at https://huggingface.co/join

2. **Create a new Space:**
   - Go to https://huggingface.co/spaces
   - Click "Create new Space"
   - Name: `rag-chatbot` (or your choice)
   - License: Apache 2.0
   - Select SDK: **Gradio**
   - Click "Create Space"

3. **Push your code to the Space:**

```bash
# Add Hugging Face as a remote
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/rag-chatbot

# Push your code
git add .
git commit -m "Initial deployment"
git push hf main
```

4. **Configure the Space:**
   - The app will use `app_github.py` as the main file
   - Dependencies from `requirements.txt` will be installed automatically
   - Your documents in `data/documents/` will be included

5. **Access your deployed app:**
   - URL: `https://huggingface.co/spaces/YOUR_USERNAME/rag-chatbot`
   - The app will automatically run `app_github.py`

**Important Notes:**
- First deployment takes 5-10 minutes to build
- Free tier has resource limitations
- App will sleep after inactivity and restart on first request

---

### 2. Render (FREE Tier Available)

Render offers free hosting for web services with some limitations.

**Steps:**

1. **Create a Render account** at https://render.com

2. **Create a new Web Service:**
   - Dashboard → New → Web Service
   - Connect your GitHub repository
   - Select branch: `main`

3. **Configure the service:**
   - **Name:** `rag-chatbot`
   - **Environment:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python app_github.py`
   - **Plan:** Free

4. **Deploy:**
   - Click "Create Web Service"
   - Render will automatically build and deploy

5. **Access your app:**
   - URL: `https://rag-chatbot-XXXX.onrender.com`

**Notes:**
- Free tier spins down after 15 minutes of inactivity
- Cold start takes ~30 seconds
- Limited to 512 MB RAM

---

### 3. Railway (FREE Tier Available)

Railway provides $5 free credits per month.

**Steps:**

1. **Create Railway account** at https://railway.app

2. **Deploy from GitHub:**
   - New Project → Deploy from GitHub repo
   - Select your repository
   - Railway auto-detects Python

3. **Configure:**
   - Add start command: `python app_github.py`
   - Environment variables (if needed)

4. **Generate domain:**
   - Settings → Generate Domain
   - Access at generated URL

---

### 4. Heroku (Paid)

Heroku no longer offers free tier, but is still a popular option.

**Steps:**

1. **Install Heroku CLI:**
```bash
# Download from https://devcenter.heroku.com/articles/heroku-cli
```

2. **Login and create app:**
```bash
heroku login
heroku create rag-chatbot-app
```

3. **Deploy:**
```bash
git push heroku main
```

4. **Open app:**
```bash
heroku open
```

**Cost:** Starts at $7/month for Eco dynos

---

## 📋 Pre-Deployment Checklist

Before deploying, ensure:

- [ ] All files are committed to git
- [ ] `requirements.txt` is up to date
- [ ] `app_github.py` works locally
- [ ] Document files are in `data/documents/`
- [ ] ChromaDB will rebuild index on first run
- [ ] `.gitignore` excludes `chroma_db/` (will be created on server)
- [ ] No API keys in code (use environment variables)

---

## 🔧 Configuration Files

### Procfile (Heroku/Render)
```
web: python app_github.py
```

### runtime.txt
```
python-3.11.7
```

### requirements.txt
Ensure all dependencies are listed with compatible versions.

---

## 🐛 Troubleshooting

### Issue: App crashes on startup
- Check logs for missing dependencies
- Verify Python version compatibility
- Ensure all imports are available

### Issue: ChromaDB errors
- ChromaDB rebuilds index on first run
- Ensure `data/documents/` exists with files
- Check disk space on hosting platform

### Issue: Out of memory
- Reduce `CHROMA_BATCH_SIZE` in code
- Use smaller embedding model
- Upgrade to paid tier with more RAM

### Issue: Slow cold starts
- Normal for free tiers
- Consider paid tier for better performance
- Use Hugging Face Spaces (faster cold starts)

---

## 🌟 Recommended: Hugging Face Spaces

For this RAG chatbot, **Hugging Face Spaces** is recommended because:

✅ **Free forever** for public spaces
✅ **Gradio optimized** - designed for Gradio apps
✅ **Fast deployment** - push to git, automatic build
✅ **Good performance** - decent RAM and CPU
✅ **Community friendly** - easy to share and discover
✅ **Persistent storage** - documents persist between restarts

---

## 📝 Next Steps After Deployment

1. **Test the deployed app** with various queries
2. **Monitor performance** and resource usage
3. **Update documents** by pushing to git
4. **Share the URL** with users
5. **Consider custom domain** (available on most platforms)

---

## 🔐 Security Notes

- Never commit API keys or secrets
- Use environment variables for sensitive data
- Keep dependencies updated for security patches
- Review platform security documentation
- Consider authentication for production use

---

## 📊 Monitoring

Most platforms provide:
- **Logs** - View application logs
- **Metrics** - CPU, memory, request stats
- **Alerts** - Get notified of issues
- **Analytics** - Track usage patterns

---

## 💡 Tips for Production

1. **Add health check endpoint** for monitoring
2. **Implement rate limiting** to prevent abuse
3. **Add caching** for frequently asked questions
4. **Monitor embedding model performance**
5. **Set up error tracking** (e.g., Sentry)
6. **Use CDN** for static assets
7. **Enable HTTPS** (usually automatic)

---

## 📚 Additional Resources

- [Hugging Face Spaces Docs](https://huggingface.co/docs/hub/spaces)
- [Render Deployment Guide](https://render.com/docs)
- [Railway Documentation](https://docs.railway.app)
- [Gradio Sharing](https://gradio.app/sharing-your-app/)

---

**Ready to deploy? Follow the Hugging Face Spaces guide above for the easiest deployment! 🚀**
