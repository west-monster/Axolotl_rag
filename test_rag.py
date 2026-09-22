import os
from src.config import settings
from src.rag_engine import RAGEngine

def test_pipeline():
    print("Starting RAG pipeline test...")
    
    # 1. Ensure we have an API KEY
    if not settings.gemini_api_key or settings.gemini_api_key == "tu_api_key_aqui":
        print("⚠️ Warning: No valid GEMINI_API_KEY configured. Test aborted.")
        return

    engine = RAGEngine()

    # 2. Clear database
    print("Clearing database...")
    engine.vector_store.clear()

    # 3. Ingest test document
    sample_path = "data/documents/sample.md"
    if not os.path.exists(sample_path):
        print(f"Error: Could not find {sample_path}")
        return

    print(f"Ingesting document {sample_path}...")
    num_chunks = engine.ingest_file(sample_path)
    print(f"Document chunked and vectorized into {num_chunks} chunks.")

    # 4. Verify insertion
    total_chunks = engine.vector_store.count()
    print(f"Total chunks in DB: {total_chunks}")
    assert total_chunks == num_chunks, "The number of chunks in the DB does not match."

    # 5. Perform query
    query = "What organs is the axolotl capable of regenerating?"
    print(f"\nPerforming query: '{query}'")
    
    result = engine.query(query)
    
    print("\nObtained response:")
    print("-" * 50)
    print(result["answer"])
    print("-" * 50)
    
    print("\nSources:")
    for doc, score in result["sources"]:
        source = doc.metadata.get("source", "unknown")
        print(f"- {os.path.basename(source)} (score: {score:.4f}): {doc.page_content[:50]}...")

    print("\nRAG Pipeline executed successfully!")

if __name__ == "__main__":
    test_pipeline()
