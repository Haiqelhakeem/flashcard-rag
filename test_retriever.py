from dotenv import load_dotenv
from utils import get_vector_db_retriever

# Load environment variables
load_dotenv()

def test_retrieval(topic: str):
    print("-" * 50)
    print(f"Testing the retriever with topic: '{topic}'")
    
    try:
        # Get the retriever from your utils file
        retriever = get_vector_db_retriever()
        
        # Invoke the retriever to find relevant documents
        retrieved_docs = retriever.invoke(topic)
        
        if not retrieved_docs:
            print("\n>>> RESULT: FAILURE! Retriever found 0 documents. <<<")
            print("This is the problem. Your vector store is either empty or isn't finding matches.")
            print("SOLUTION: Rerun `create_vectorstore.py` and check its output to ensure it's processing your PDFs.\n")
        else:
            print(f"\n>>> RESULT: SUCCESS! Retriever found {len(retrieved_docs)} documents. <<<")
            for i, doc in enumerate(retrieved_docs):
                print(f"\n--- Doc {i+1} (Source: {doc.metadata.get('source', 'N/A')}, Page: {doc.metadata.get('page', 'N/A')}) ---")
                print(f"{doc.page_content[:400]}...")
            print("\nIf you see this, your retriever is working correctly. The error is likely with the LLM API call.")

    except Exception as e:
        print(f"\nAn error occurred while testing the retriever: {e}")

if __name__ == "__main__":
    test_retrieval("dna")
    test_retrieval("reproduction")
    test_retrieval("evolution")
    test_retrieval("photosynthesis")
    test_retrieval("respiration")