import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables from your .env file
load_dotenv()

# Define the path to the vector store you created with create_vectorstore.py
VECTORSTORE_PATH = "./faiss_index"

def get_vector_db_retriever():
    """
    Loads the persistent FAISS vector store from the local file system
    and returns it as a retriever.
    """
    # Check if the vector store path exists
    if not os.path.exists(VECTORSTORE_PATH):
        raise FileNotFoundError(
            f"Vector store not found at '{VECTORSTORE_PATH}'. "
            "Please run the `create_vectorstore.py` script first."
        )

    # Initialize the same embedding model used when creating the vector store
    embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=os.environ.get("GEMINI_API_KEY")
        )

    # Load the vector store from the local folder
    # allow_dangerous_deserialization is needed for FAISS with pickle
    db = FAISS.load_local(
        VECTORSTORE_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True
    )

    # Return the vector store as a retriever
    # This will be used to find relevant documents for your RAG chain
    return db.as_retriever(search_kwargs={'score_threshold': 0.7})