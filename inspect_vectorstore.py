import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Muat environment variables
load_dotenv()

# Definisikan path
VECTORSTORE_PATH = "./faiss_index"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def inspect_vector_store(search_term: str):
    """
    Memuat vector store yang ada dan mencari keyword di dalam konten teksnya.
    """
    print("\n" + "=" * 60)
    print(f"🔬 Menginspeksi isi vector store untuk kata kunci: '{search_term}'")
    print("=" * 60)

    try:
        # 1. Pastikan vector store ada
        if not os.path.exists(VECTORSTORE_PATH):
            print(f"❌ ERROR: Direktori vector store tidak ditemukan di '{VECTORSTORE_PATH}'.")
            print("Pastikan Anda sudah menjalankan `create_vectorstore.py` setidaknya sekali.")
            return

        # 2. Muat model embedding (diperlukan untuk memuat FAISS)
        print("Menginisialisasi model embedding...")
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'}
        )

        # 3. Muat vector store FAISS dari disk
        print(f"Memuat vector store dari '{VECTORSTORE_PATH}'...")
        db = FAISS.load_local(
            VECTORSTORE_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        print("Vector store berhasil dimuat.")

        # 4. Akses dan cari di dalam 'docstore'
        # Di sinilah semua potongan teks asli disimpan
        print(f"Mencari '{search_term}' di dalam {len(db.docstore._dict)} potongan teks yang tersimpan...")
        
        found_count = 0
        for doc_id, doc in db.docstore._dict.items():
            if search_term.lower() in doc.page_content.lower():
                found_count += 1
                print("\n" + "-" * 50)
                print(f"✅ DITEMUKAN di Dokumen #{found_count}")
                print(f"   Sumber: {doc.metadata.get('source', 'N/A')}")
                print(f"   Halaman: {doc.metadata.get('page', 'N/A')}")
                print(f"   Isi Chunk:\n   '{doc.page_content[:500]}...'")
        
        print("\n" + "=" * 60)
        if found_count > 0:
            print(f">>> ✅ KESIMPULAN: Ditemukan {found_count} chunk yang mengandung '{search_term}'.")
        else:
            print(f">>> ❌ KESIMPULAN: TIDAK DITEMUKAN satu pun chunk yang mengandung '{search_term}'.")

    except Exception as e:
        print(f"\nTerjadi error: {e}")

if __name__ == "__main__":
    keyword = input("Masukkan kata kunci untuk dicari di dalam vector store (contoh: viruses): ")
    if keyword:
        inspect_vector_store(keyword)
    else:
        print("Tidak ada kata kunci yang dimasukkan. Program berhenti.")