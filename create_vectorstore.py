import os
import re
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables from .env file
load_dotenv()

# Define paths
DOCUMENTS_PATH = "./documents"
VECTORSTORE_PATH = "./faiss_index"

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

    # Step 2: Clean and filter document pages
    print("Cleaning and filtering document pages...")
    cleaned_docs = []
    for doc in docs:
        content = re.sub(r'\s+', ' ', doc.page_content).strip()
        if len(content) > 200: # Your great filter for useful pages
            doc.page_content = content
            cleaned_docs.append(doc)
            
    print(f"Filtered down to {len(cleaned_docs)} useful pages.")
    
    # Step 3: Split documents
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    # CHANGE 3: We now split the 'cleaned_docs', not the original 'docs'
    texts = text_splitter.split_documents(cleaned_docs)
    print(f"Split documents into {len(texts)} chunks.")

    # Step 4: Initialize the FREE, LOCAL embedding model
    print("Initializing local embedding model (this may download the model on first run)...")
    # This model runs on your computer, no API key or internet needed after download
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'}
    )

    # CHANGE 4: Simplified vector store creation (No batching needed)
    # Because the model is local, we don't have rate limits. We can process everything at once.
    print(f"Creating vector store from {len(texts)} chunks...")
    db = FAISS.from_documents(texts, embeddings)
    
    # Step 5: Save the final vector store
    print(f"Saving vector store to '{VECTORSTORE_PATH}'...")
    db.save_local(VECTORSTORE_PATH)
    print("\nVector store created and saved successfully!")

if __name__ == "__main__":
    main()