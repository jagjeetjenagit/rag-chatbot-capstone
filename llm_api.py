"""
LLM API integrations for RAG Chatbot.
Handles communication with OpenAI, Google Gemini, and Ollama.
"""

import json
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# OpenAI API Integration
# ============================================================================

def call_openai_api(
    prompt: str,
    api_key: str,
    model: str = "gpt-3.5-turbo",
    max_tokens: int = 512,
    temperature: float = 0.7
) -> Optional[str]:
    """
    Call OpenAI API for answer generation.
    
    Args:
        prompt: The prompt to send to OpenAI
        api_key: OpenAI API key
        model: Model name (gpt-3.5-turbo, gpt-4, etc.)
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature (0-2)
        
    Returns:
        Generated text or None if error
    """
    try:
        import openai
        
        client = openai.OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return response.choices[0].message.content
        
    except ImportError:
        logger.error("openai package not installed. Run: pip install openai")
        return None
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return None


# ============================================================================
# Google Gemini API Integration
# ============================================================================

def call_gemini_api(
    prompt: str,
    api_key: str,
    model: str = "gemini-pro",
    max_tokens: int = 512,
    temperature: float = 0.7
) -> Optional[str]:
    """
    Call Google Gemini API for answer generation using REST API.
    
    Args:
        prompt: The prompt to send to Gemini
        api_key: Google API key
        model: Model name (gemini-pro, gemini-1.5-flash, etc.)
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature (0-2)
        
    Returns:
        Generated text or None if error
    """
    try:
        import requests
        
        # Use the REST API directly (v1 instead of v1beta)
        url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent"
        
        headers = {
            "Content-Type": "application/json"
        }
        
        data = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature
            }
        }
        
        response = requests.post(
            f"{url}?key={api_key}",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if "candidates" in result and len(result["candidates"]) > 0:
                return result["candidates"][0]["content"]["parts"][0]["text"]
        else:
            logger.error(f"Gemini API error: {response.status_code} - {response.text}")
        
        return None
        
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return None


# ============================================================================
# Ollama API Integration (Local LLM)
# ============================================================================

def call_ollama_api(
    prompt: str,
    base_url: str = "http://localhost:11434",
    model: str = "llama2"
) -> Optional[str]:
    """
    Call Ollama API for answer generation (local LLM).
    
    Args:
        prompt: The prompt to send to Ollama
        base_url: Ollama server URL
        model: Model name (llama2, mistral, etc.)
        
    Returns:
        Generated text or None if error
    """
    try:
        import requests
        
        response = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()["response"]
        else:
            logger.error(f"Ollama API error: {response.status_code}")
            return None
            
    except Exception as e:
        logger.error(f"Ollama API error: {e}")
        return None


# ============================================================================
# Unified LLM Call
# ============================================================================

def call_llm(
    prompt: str,
    provider: str = "auto",
    config: Optional[Dict] = None
) -> Optional[str]:
    """
    Call LLM with automatic provider selection.
    
    Args:
        prompt: The prompt to send
        provider: "openai", "google", "ollama", or "auto"
        config: Configuration dictionary (from config.py)
        
    Returns:
        Generated text or None if all providers fail
    """
    from config import get_llm_config, is_api_configured
    
    if config is None:
        config = {}
    
    # Try providers in order if auto
    if provider == "auto":
        providers = ["openai", "google", "ollama"]
    else:
        providers = [provider]
    
    for prov in providers:
        if not is_api_configured(prov):
            logger.debug(f"Skipping {prov}: not configured")
            continue
        
        logger.info(f"Trying {prov} API...")
        llm_config = get_llm_config(prov)
        llm_config.update(config)  # Override with custom config
        
        if prov == "openai":
            result = call_openai_api(prompt, **llm_config)
        elif prov == "google":
            result = call_gemini_api(prompt, **llm_config)
        elif prov == "ollama":
            # Ollama doesn't use max_tokens, so remove it
            ollama_config = {k: v for k, v in llm_config.items() if k != 'max_tokens' and k != 'temperature'}
            result = call_ollama_api(prompt, **ollama_config)
        else:
            continue
        
        if result:
            logger.info(f"✅ {prov} API successful")
            return result
    
    logger.warning("All LLM providers failed or not configured")
    return None


# ============================================================================
# Test Functions
# ============================================================================

if __name__ == "__main__":
    """Test API integrations."""
    import config
    
    print("=" * 80)
    print("TESTING LLM API INTEGRATIONS")
    print("=" * 80)
    
    test_prompt = "What is machine learning? Answer in one sentence."
    
    # Test OpenAI
    if config.is_api_configured("openai"):
        print("\n🧪 Testing OpenAI...")
        result = call_openai_api(
            test_prompt,
            api_key=config.get_api_key("openai"),
            model=config.OPENAI_MODEL,
            max_tokens=50
        )
        if result:
            print(f"✅ OpenAI Response: {result}")
        else:
            print("❌ OpenAI failed")
    else:
        print("\n⏭️  Skipping OpenAI (not configured)")
    
    # Test Google Gemini
    if config.is_api_configured("google"):
        print("\n🧪 Testing Google Gemini...")
        result = call_gemini_api(
            test_prompt,
            api_key=config.get_api_key("google"),
            model=config.GOOGLE_MODEL,
            max_tokens=50
        )
        if result:
            print(f"✅ Gemini Response: {result}")
        else:
            print("❌ Gemini failed")
    else:
        print("\n⏭️  Skipping Google Gemini (not configured)")
    
    # Test Ollama
    print("\n🧪 Testing Ollama (local)...")
    result = call_ollama_api(
        test_prompt,
        base_url=config.OLLAMA_BASE_URL,
        model=config.OLLAMA_MODEL
    )
    if result:
        print(f"✅ Ollama Response: {result}")
    else:
        print("❌ Ollama failed (is Ollama running?)")
    
    # Test unified call
    print("\n🧪 Testing unified LLM call (auto)...")
    result = call_llm(test_prompt, provider="auto")
    if result:
        print(f"✅ Auto Response: {result}")
    else:
        print("❌ All providers failed")
    
    print("\n" + "=" * 80)
