import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

VECTORSTORE_PATH = "./faiss_index"

def get_vector_db_retriever():
    """
    Loads the persistent FAISS vector store.
    """
    if not os.path.exists(VECTORSTORE_PATH):
        raise FileNotFoundError(
            f"Vector store not found at '{VECTORSTORE_PATH}'."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'} 
    )
    
    db = FAISS.load_local(
        VECTORSTORE_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True
    )

    # RESTORED: Simple retriever, k=8 (Good balance of speed and context)
    return db.as_retriever(
        search_type="similarity", 
        search_kwargs={'k': 10} 
    )