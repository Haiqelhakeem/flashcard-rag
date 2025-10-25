import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
# CHANGE 1: Import the new embedding class
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

VECTORSTORE_PATH = "./faiss_index"

def get_vector_db_retriever():
    """
    Loads the persistent FAISS vector store from the local file system
    and returns it as a retriever.
    """
    if not os.path.exists(VECTORSTORE_PATH):
        raise FileNotFoundError(
            f"Vector store not found at '{VECTORSTORE_PATH}'. "
            "Please run the `create_vectorstore.py` script first."
        )

    # CHANGE 2: Use the free, local Hugging Face model instead of Google's
    # This model ('all-MiniLM-L6-v2') is small, fast, and effective.
    # It will be downloaded automatically the first time you run it.
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'} # Use CPU for broad compatibility
    )
    
    db = FAISS.load_local(
        VECTORSTORE_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True
    )

    return db.as_retriever(
        search_type="mmr",
        search_kwargs={'score_threshold': 0.6}
        )