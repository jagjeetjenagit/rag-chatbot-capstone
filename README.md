---
title: RAG Chatbot - Document Q&A
emoji: 📚
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 3.50.2
app_file: app_github.py
pinned: false
license: mit
---

# 📚 RAG Chatbot - Intelligent Document Q&A System

Ask questions about our comprehensive document collection and get accurate answers with source citations!

## 🌟 Features

- **20+ Documents Indexed**: Technical guides and business data
- **Smart Search**: Semantic search using ChromaDB vector database
- **Source Citations**: Every answer includes references
- **Interactive UI**: Adjustable parameters for customized responses

## 📖 What You Can Ask

### Technical Topics
- Machine Learning fundamentals
- Python programming concepts
- Artificial Intelligence overview
- Deep Learning techniques
- Data Science principles

### Business Data
- Employee salaries and compensation
- Company financial performance
- Department metrics and KPIs
- Training and development programs
- HR policies and benefits
- Strategic initiatives and OKRs

## 🎯 Example Queries

```
What is machine learning?
What is the average salary for engineers?
How much profit did the company make in 2025?
Which department contributes most to revenue?
What are the company's training investments?
```

## 🔧 How It Works

1. **Your Question** → Entered in natural language
2. **Vector Search** → Finds relevant document chunks
3. **Context Assembly** → Gathers top matching sources
4. **Answer Generation** → Creates response from context
5. **Source Attribution** → Shows where info came from

## 📊 System Details

- **Documents**: 20 comprehensive files
- **Indexed Chunks**: 139 total
- **Embedding Model**: all-MiniLM-L6-v2
- **Vector Database**: ChromaDB
- **Framework**: Gradio 3.50.2

## 🎨 Customization

Use the interface controls to:
- **Adjust Sources (1-10)**: More sources = broader context
- **Temperature (0.0-1.0)**: Control response creativity
  - Lower (0.0-0.3): Focused and precise
  - Higher (0.7-1.0): Creative and diverse

## 🚀 Try It Now!

Enter a question below and explore our document collection!

---

**Built with:** ChromaDB • Sentence Transformers • Gradio  
**GitHub:** [rag-chatbot-capstone](https://github.com/jagjeetjenagit/rag-chatbot-capstone)
