import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

VECTORSTORE_PATH = "./faiss_index"
# The model you BELIEVE you used
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
def diagnose():
    print(f"🕵️ Diagnosing Vector Store at: {VECTORSTORE_PATH}")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={'device': 'cpu'} 
    )

    try:
        db = FAISS.load_local(
            VECTORSTORE_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        print(f"❌ Error loading DB: {e}")
        return

    # Use a term you KNOW is in the database
    test_query = "sel" # Or "DNA", "mitosis", etc.
    
    print(f"\n🧪 Test Query: '{test_query}'")
    
    # Get RAW L2 Distance (no conversion)
    # This shows the true mathematical distance
    docs_and_scores = db.similarity_search_with_score(test_query, k=3)
    
    print("\n--- RAW L2 SCORES (Lower is Better) ---")
    for doc, score in docs_and_scores:
        print(f"Score: {score:.4f} | Content: {doc.page_content[:50]}...")

    print("\n--- DIAGNOSIS ---")
    first_score = docs_and_scores[0][1]
    
    if first_score > 20:
        print("⚠️  Scores are VERY HIGH (> 20).")
        print("👉 Conclusion: Your vectors are likely NOT normalized.")
        print("👉 FIX: You cannot use 'score_threshold' easily. Use 'top_k' only, OR set a very high threshold (e.g., 300).")
    elif first_score < 1.5:
        print("✅ Scores are LOW (0 to 1.5).")
        print("👉 Conclusion: Your vectors are normalized L2.")
        print("👉 FIX: Set your threshold logic to: 'score < 0.8' (Pass) vs 'score > 0.8' (Fail).")
    else:
        print("⚠️  Scores are WEIRD (negative or huge).")
        print("👉 Conclusion: Potential Model Mismatch.")

if __name__ == "__main__":
    diagnose()