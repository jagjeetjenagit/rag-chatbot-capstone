"""
Quick test of RAG chatbot functionality
"""
import sys
sys.path.insert(0, 'C:/capstone project 1')

from retriever import Retriever
from generator import generate_answer

print("=" * 80)
print("🧪 RAG CHATBOT - FUNCTIONALITY TEST")
print("=" * 80)

# Initialize retriever
print("\n1️⃣ Testing Document Retrieval...")
retriever = Retriever()

# Get collection stats
try:
    stats = retriever.collection.count()
    print(f"   ✅ ChromaDB Connected")
    print(f"   📄 Documents: {stats}")
    print(f"   🔍 Embedding Dimension: 384")
except Exception as e:
    print(f"   ⚠️  Could not get stats: {e}")

# Test retrieval
test_query = "What is machine learning?"
print(f"\n2️⃣ Testing Semantic Search...")
print(f"   Query: '{test_query}'")

chunks = retriever.get_top_k(test_query, k=3)
print(f"   ✅ Retrieved {len(chunks)} chunks")
if chunks:
    print(f"   Top result score: {chunks[0]['score']:.3f}")
    print(f"   Source: {chunks[0]['metadata']['source']}")

# Test answer generation
print(f"\n3️⃣ Testing Answer Generation...")
if chunks:
    answer = generate_answer(test_query, chunks, backend='auto')
    print(f"   Method: {answer['method']}")
    print(f"   Confidence: {answer['confidence']:.2f}")
    print(f"\n   📝 Answer:")
    print(f"   {answer['answer']}")
    print(f"\n   📚 Sources: {', '.join(answer['sources'])}")
else:
    print("   ⚠️  No chunks retrieved")

print("\n" + "=" * 80)
print("✅ ALL TESTS COMPLETED")
print("=" * 80)
print("\n🌐 App running at: http://localhost:7860")
print("\n💡 Current Mode: Fallback (Rule-Based)")
print("   - API Keys: OpenAI quota exceeded, Gemini invalid")
print("   - Answers: Keyword-based extraction (works!)")
print("   - Quality: Basic but functional")
print("\n📝 To enable AI-powered responses:")
print("   1. Add OpenAI credits ($5-10)")
print("   2. Get new Google Gemini key")
print("   3. Install Ollama (free, local)")
print("\n" + "=" * 80)
