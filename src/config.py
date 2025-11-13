"""
Configuration module for RAG Chatbot.
Stores all configurable parameters and settings.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
CHROMA_DB_DIR.mkdir(exist_ok=True)

# Text Chunking Configuration
CHUNK_SIZE_MIN = 500  # Minimum characters per chunk
CHUNK_SIZE_MAX = 800  # Maximum characters per chunk
CHUNK_OVERLAP_PERCENT = 10  # 10% overlap between chunks

# Embedding Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # Dimension for all-MiniLM-L6-v2

# Vector Store Configuration
CHROMA_COLLECTION_NAME = "rag_documents"
TOP_K_RETRIEVAL = 5  # Number of top chunks to retrieve (3-5 as per spec)

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # Options: "openai", "google"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-pro")

# LLM Generation Parameters
MAX_TOKENS = 1000
TEMPERATURE = 0.7

# Gradio UI Configuration
GRADIO_SHARE = False  # Set to True to create public link
GRADIO_SERVER_PORT = 7860
GRADIO_SERVER_NAME = "127.0.0.1"

# System Prompts
SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on the provided context.
Always cite the source document when providing information.
If the context doesn't contain enough information to answer the question, say so clearly.
Be concise and accurate in your responses."""

RAG_PROMPT_TEMPLATE = """Context from documents:
{context}

Question: {question}

Please provide a clear answer based on the context above. Include source citations in your response.
If the context doesn't contain relevant information, state that clearly."""

# Document Processing
SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".docx"]
MAX_FILE_SIZE_MB = 50  # Maximum file size in MB
