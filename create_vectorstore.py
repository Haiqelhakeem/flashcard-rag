import os
import re
import time # Import the time module to add delays
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables from .env file
load_dotenv()

# Define paths
DOCUMENTS_PATH = "./documents"
VECTORSTORE_PATH = "./faiss_index"

# --- NEW: Helper function to create batches ---
def create_batches(data, batch_size):
    """Yield successive n-sized chunks from a list."""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def main():
    """
    Main function to create and save the vector store.
    """
    print("Starting the creation of the vector store...")

    # Step 1: Load documents
    print(f"Loading documents from '{DOCUMENTS_PATH}'...")
    loader = DirectoryLoader(
        DOCUMENTS_PATH, 
        glob="**/*.pdf", 
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    docs = loader.load()
    if not docs:
        print("No documents found.")
        return
    print(f"Loaded {len(docs)} documents.")

    # Step 2: Clean and pre-process text
    print("Cleaning and pre-processing document text...")
    for doc in docs:
        doc.page_content = re.sub(r'\s+', ' ', doc.page_content).strip()
    print("Text cleaning complete.")

    # Step 3: Split documents
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(docs)
    print(f"Split documents into {len(texts)} chunks.")

    # Step 4: Initialize embedding model
    print("Initializing embedding model...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.environ.get("GEMINI_API_KEY")
    )

    # --- MODIFIED: Process in batches ---
    print("Processing embeddings in batches...")
    batch_size = 100  # A safe batch size for the API
    text_batches = list(create_batches(texts, batch_size))
    
    db = None
    for i, batch in enumerate(text_batches):
        print(f"Processing batch {i+1}/{len(text_batches)}...")
        if i == 0:
            # For the first batch, create the vector store
            db = FAISS.from_documents(batch, embeddings)
        else:
            # For subsequent batches, add them to the existing store
            db.add_documents(batch)
        
        # Add a small delay between batches to avoid hitting rate limits
        time.sleep(1) 

    # Step 5: Save the final vector store
    if db:
        print(f"Saving vector store to '{VECTORSTORE_PATH}'...")
        db.save_local(VECTORSTORE_PATH)
        print("\nVector store created and saved successfully!")
    else:
        print("Could not create the vector store.")

if __name__ == "__main__":
    main()