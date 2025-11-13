"""
Integration Demo: Full RAG Pipeline
Shows complete flow: Query → Retrieve → Generate
"""
from retriever import get_top_k
from generator import generate_answer, format_answer_for_display

print("=" * 80)
print("FULL RAG PIPELINE DEMO")
print("=" * 80)

# Test query
query = "What is chunking and why is it important?"

print(f"\n📝 Query: {query}")
print("\n" + "-" * 80)

# Step 1: Retrieve relevant chunks
print("\n🔍 STEP 1: Retrieving relevant chunks...")
chunks = get_top_k(query, k=3)
print(f"   Retrieved {len(chunks)} chunks")

if chunks:
    print("\n   Top results:")
    for i, chunk in enumerate(chunks, 1):
        print(f"   [{i}] Score: {chunk['score']:.3f} | Source: {chunk['metadata']['source']}")

# Step 2: Generate answer
print("\n🤖 STEP 2: Generating answer...")
result = generate_answer(query, chunks, max_tokens=512)

# Step 3: Display result
print("\n" + "=" * 80)
print("ANSWER")
print("=" * 80)
print(format_answer_for_display(result))

print("\n" + "=" * 80)
print("Pipeline complete! ✅")
print("=" * 80)
