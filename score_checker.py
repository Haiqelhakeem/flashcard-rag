import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
VECTORSTORE_PATH = "./faiss_index"
# IMPORTANT: This must match the model used in get_vector_db_retriever
# (Your provided code used 'paraphrase-multilingual-MiniLM-L12-v2')
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CURRENT_THRESHOLD = 0.6

def check_retrieval_scores(query: str):
    """
    Performs a search and prints the raw distance and the calculated relevance score.
    """
    print(f"\n🔍 Analyzing Query: '{query}'")
    print("=" * 60)

    if not os.path.exists(VECTORSTORE_PATH):
        print(f"❌ Error: Vector store not found at {VECTORSTORE_PATH}")
        return

    # 1. Initialize Embeddings
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'} 
    )

    # 2. Load Vector Store
    try:
        db = FAISS.load_local(
            VECTORSTORE_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        print(f"❌ Failed to load FAISS index: {e}")
        return

    # 3. Perform Search with Relevance Scores
    # specific method that converts L2 distance to a 0-1 scale
    results = db.similarity_search_with_relevance_scores(query, k=5)

    if not results:
        print("No results found.")
        return

    print(f"\nResults (Top 5) | Threshold to beat: {CURRENT_THRESHOLD}")
    print("-" * 60)

    for i, (doc, score) in enumerate(results):
        # Determine pass/fail based on your threshold
        status = "✅ PASS" if score >= CURRENT_THRESHOLD else "❌ FAIL"
        
        print(f"Rank {i+1}: {status}")
        print(f"   📊 Similarity Score: {score:.4f} (Higher is better)")
        # FAISS L2 Distance roughly equals: (1 / score) - 1 (approximation varies by implementation)
        
        print(f"   📄 Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"   📝 Content Preview: {doc.page_content[:150].replace('\n', ' ')}...")
        print("-" * 60)

if __name__ == "__main__":
    print("--- RAG Score Checker ---")
    while True:
        user_input = input("\nEnter a keyword to check scores (or 'q' to quit): ")
        if user_input.lower() in ['q', 'quit', 'exit']:
            break
        if user_input.strip():
            check_retrieval_scores(user_input)