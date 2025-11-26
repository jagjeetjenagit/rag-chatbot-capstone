"""
Configuration file for RAG Chatbot.
Loads API keys from .env file.
"""

import os
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()  # Loads .env file from current directory
except ImportError:
    print("Warning: python-dotenv not installed. Run: pip install python-dotenv")
    print("Continuing without .env file...")

# ============================================================================
# API CONFIGURATION
# ============================================================================

# LLM Provider Selection
# Loaded from .env file, defaults to "auto"
# Options: "openai", "google", "ollama", "auto" (tries in order), or "fallback" (rule-based only)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto")

# OpenAI API Configuration
# Get your API key from: https://platform.openai.com/api-keys
# Add to .env file: OPENAI_API_KEY=sk-your-key-here
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "512"))
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))

# Google Gemini API Configuration
# Get your API key from: https://makersuite.google.com/app/apikey
# Add to .env file: GOOGLE_API_KEY=your-key-here
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-pro")  # gemini-pro is stable
GOOGLE_MAX_TOKENS = int(os.getenv("GOOGLE_MAX_TOKENS", "512"))
GOOGLE_TEMPERATURE = float(os.getenv("GOOGLE_TEMPERATURE", "0.7"))

# Ollama Configuration (Local LLM)
# Install Ollama from: https://ollama.ai
# Add to .env file: OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")

# ============================================================================
# RAG SYSTEM CONFIGURATION
# ============================================================================

# Document Processing
DOCUMENTS_DIR = "data/documents"
CHUNK_SIZE_MIN = 500
CHUNK_SIZE_MAX = 800
CHUNK_OVERLAP_PERCENT = 10

# Vector Database
CHROMA_DB_PATH = "./chroma_db"
CHROMA_COLLECTION_NAME = "capstone_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # sentence-transformers model

# Retrieval
SIMILARITY_THRESHOLD = 0.2  # Threshold for semantic search fallback to keyword search
TOP_K_RETRIEVAL = 4  # Number of chunks to retrieve

# Gradio UI
GRADIO_PORT = 7860
GRADIO_SHARE = False  # Set to True to create a public link

# ============================================================================
# LOGGING
# ============================================================================

LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_api_key(provider: str) -> str:
    """
    Get API key for the specified provider.
    
    Args:
        provider: "openai", "google", or "ollama"
        
    Returns:
        API key string or empty string if not configured
    """
    if provider == "openai":
        return OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")
    elif provider == "google":
        return GOOGLE_API_KEY or os.getenv("GOOGLE_API_KEY", "")
    elif provider == "ollama":
        return ""  # Ollama doesn't need API key
    return ""


def is_api_configured(provider: str) -> bool:
    """
    Check if API is properly configured for the provider.
    
    Args:
        provider: "openai", "google", or "ollama"
        
    Returns:
        True if configured, False otherwise
    """
    if provider == "openai":
        return bool(get_api_key("openai"))
    elif provider == "google":
        return bool(get_api_key("google"))
    elif provider == "ollama":
        return True  # Ollama is local, no key needed
    return False


def get_llm_config(provider: str) -> dict:
    """
    Get LLM configuration for the specified provider.
    
    Args:
        provider: "openai", "google", or "ollama"
        
    Returns:
        Dictionary with provider configuration
    """
    configs = {
        "openai": {
            "api_key": get_api_key("openai"),
            "model": OPENAI_MODEL,
            "max_tokens": OPENAI_MAX_TOKENS,
            "temperature": OPENAI_TEMPERATURE,
        },
        "google": {
            "api_key": get_api_key("google"),
            "model": GOOGLE_MODEL,
            "max_tokens": GOOGLE_MAX_TOKENS,
            "temperature": GOOGLE_TEMPERATURE,
        },
        "ollama": {
            "base_url": OLLAMA_BASE_URL,
            "model": OLLAMA_MODEL,
        }
    }
    return configs.get(provider, {})


def validate_config():
    """Validate configuration and print warnings if needed."""
    warnings = []
    
    if LLM_PROVIDER == "openai" and not is_api_configured("openai"):
        warnings.append("⚠️  OpenAI API key not configured. Add it to config.py")
    
    if LLM_PROVIDER == "google" and not is_api_configured("google"):
        warnings.append("⚠️  Google API key not configured. Add it to config.py")
    
    if LLM_PROVIDER == "auto" and not any([is_api_configured("openai"), is_api_configured("google")]):
        warnings.append("ℹ️  No API keys configured. Using rule-based fallback for answer generation.")
    
    if not Path(DOCUMENTS_DIR).exists():
        warnings.append(f"⚠️  Documents directory not found: {DOCUMENTS_DIR}")
    
    return warnings


if __name__ == "__main__":
    """Test configuration."""
    print("=" * 80)
    print("RAG CHATBOT CONFIGURATION")
    print("=" * 80)
    print(f"\n📁 Documents Directory: {DOCUMENTS_DIR}")
    print(f"🔧 LLM Provider: {LLM_PROVIDER}")
    print(f"🗄️  ChromaDB Path: {CHROMA_DB_PATH}")
    print(f"🔍 Embedding Model: {EMBEDDING_MODEL}")
    print(f"📊 Top-K Retrieval: {TOP_K_RETRIEVAL}")
    print(f"🌐 Gradio Port: {GRADIO_PORT}")
    
    print("\n" + "=" * 80)
    print("API STATUS")
    print("=" * 80)
    print(f"OpenAI: {'✅ Configured' if is_api_configured('openai') else '❌ Not configured'}")
    print(f"Google: {'✅ Configured' if is_api_configured('google') else '❌ Not configured'}")
    print(f"Ollama: {'✅ Available (local)' if is_api_configured('ollama') else '❌ Not available'}")
    
    print("\n" + "=" * 80)
    print("VALIDATION")
    print("=" * 80)
    warnings = validate_config()
    if warnings:
        for warning in warnings:
            print(warning)
    else:
        print("✅ All configuration valid!")
    
    print("\n" + "=" * 80)
