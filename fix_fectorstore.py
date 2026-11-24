import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# --- CONFIGURATION ---
DATA_PATH = "./documents"  # Ensure your PDFs are here
NEW_DB_PATH = "./faiss_index_fixed" # <--- NEW FOLDER NAME
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def rebuild_clean_db():
    print(f"🚀 Building NEW database at: {NEW_DB_PATH}")
    
    # 1. Load & Split
    print("📂 Loading docs...")
    loader = DirectoryLoader(DATA_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    print(f"✅ Prepared {len(texts)} chunks.")

    # 2. Embed with the CORRECT Model
    print("🧠 Embedding...")
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True} # <--- CRITICAL FIX: Force Normalization
    )

    # 3. Save
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(NEW_DB_PATH)
    print("🎉 Done! You now have a clean database.")

if __name__ == "__main__":
    rebuild_clean_db()