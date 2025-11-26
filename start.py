"""
Quick Start Script for RAG Chatbot
Run this to launch the application with proper configuration.
"""

import sys
import os
from pathlib import Path

def main():
    # Make sure we're using the virtual environment if it exists
    venv_python = Path("C:/capstone project 1/.venv/Scripts/python.exe")
    current_python = sys.executable
    
    if venv_python.exists() and str(venv_python) not in current_python:
        print(f"⚠️  Please run with virtual environment:")
        print(f'   & "{venv_python}" start.py')
        print()
        print("Or activate the venv first:")
        print("   .venv\\Scripts\\Activate.ps1")
        sys.exit(1)
    
    print("=" * 80)
    print("🚀 RAG CHATBOT - QUICK START")
    print("=" * 80)
    
    # Check configuration
    try:
        import config
        print("\n✅ Configuration loaded successfully")
        
        # Validate and show warnings
        warnings = config.validate_config()
        if warnings:
            print("\n⚠️  Configuration Warnings:")
            for warning in warnings:
                print(f"  {warning}")
        
        # Show current settings
        print(f"\n📋 Current Settings:")
        print(f"  LLM Provider: {config.LLM_PROVIDER}")
        print(f"  Documents Dir: {config.DOCUMENTS_DIR}")
        print(f"  Gradio Port: {config.GRADIO_PORT}")
        
        # Check API status
        print(f"\n🔑 API Status:")
        if config.is_api_configured("openai"):
            print(f"  OpenAI: ✅ Configured ({config.OPENAI_MODEL})")
        else:
            print(f"  OpenAI: ❌ Not configured")
        
        if config.is_api_configured("google"):
            print(f"  Google: ✅ Configured ({config.GOOGLE_MODEL})")
        else:
            print(f"  Google: ❌ Not configured")
        
        if config.is_api_configured("ollama"):
            print(f"  Ollama: ✅ Available (local)")
        else:
            print(f"  Ollama: ❌ Not available")
        
        # Check if any API is configured
        if config.LLM_PROVIDER == "auto":
            has_api = any([
                config.is_api_configured("openai"),
                config.is_api_configured("google"),
                config.is_api_configured("ollama")
            ])
            
            if not has_api:
                print("\n💡 TIP: No API configured. App will use rule-based fallback.")
                print("   For better answers, configure an API key in config.py")
                print("   See API_SETUP.md for instructions.")
    
    except ImportError:
        print("\n❌ config.py not found!")
        print("   Make sure you're in the correct directory.")
        sys.exit(1)
    
    # Launch the app
    print("\n" + "=" * 80)
    print("🎯 LAUNCHING RAG CHATBOT...")
    print("=" * 80)
    print("\nThis may take a moment to:")
    print("  1. Load the embedding model (5-10 seconds)")
    print("  2. Initialize ChromaDB")
    print("  3. Start Gradio server")
    print("\nPlease wait...")
    print("-" * 80)
    
    try:
        import rag_app
        rag_app.main()
    except KeyboardInterrupt:
        print("\n\n👋 Chatbot stopped by user")
    except Exception as e:
        print(f"\n\n❌ Error starting app: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure all dependencies are installed: pip install -r requirements.txt")
        print("  2. Check that port 7860 is not in use")
        print("  3. Run: python config.py to validate configuration")
        sys.exit(1)


if __name__ == "__main__":
    main()
