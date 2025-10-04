# SANITY CHECK: Pastikan Anda melihat pesan ini saat menjalankan skrip.
print("--- RUNNING LATEST VERSION OF TEST_RETRIEVER.PY ---")
print(f"Current Time: {__import__('datetime').datetime.now()}")

import sys
from dotenv import load_dotenv
from utils import get_vector_db_retriever

# Muat environment variables
load_dotenv()

def test_retrieval(topic: str):
    """
    Menjalankan tes retrieval untuk sebuah topik dan mencetak hasilnya.
    """
    print("\n" + "=" * 60)
    print(f"🔬 Testing retriever for topic: '{topic}'")
    print("=" * 60)
    
    try:
        # Dapatkan retriever dari file utils Anda
        retriever = get_vector_db_retriever()
        
        # Panggil retriever untuk menemukan dokumen yang relevan
        retrieved_docs = retriever.invoke(topic)
        
        if not retrieved_docs:
            print("\n>>> ❌ RESULT: FAILURE! No documents found for this topic. <<<")
            print("\nPossible Causes & Solutions:")
            print("1. Content Gap: Topik ini mungkin tidak ada di dalam PDF sumber Anda.")
            print("   -> ACTION: Buka PDF Anda dan cari (Ctrl+F) topik tersebut secara manual.")
            print("2. Filtering Issue: Konten mungkin ada di halaman yang terfilter keluar (misalnya, teks < 200 karakter).")
            print("   -> ACTION: Periksa logika pembersihan/filter di `create_vectorstore.py`.")
            print("3. Stale Index: Vector store Anda mungkin belum diperbarui setelah ada perubahan dokumen.")
            print("   -> ACTION: Jalankan kembali `create_vectorstore.py`.")
        else:
            print(f"\n>>> ✅ RESULT: SUCCESS! Found {len(retrieved_docs)} relevant document(s). <<<")
            for i, doc in enumerate(retrieved_docs):
                source = doc.metadata.get('source', 'N/A')
                page = doc.metadata.get('page', 'N/A')
                print(f"\n--- Document {i+1} | Source: {source}, Page: {page} ---")
                print(f"Content: {doc.page_content[:400]}...")

    except FileNotFoundError as e:
        print(f"\n>>> ❌ ERROR: Could not find the vector store! <<<")
        print(f"Details: {e}")
        print("SOLUTION: Pastikan Anda telah berhasil menjalankan skrip `create_vectorstore.py` terlebih dahulu.")
    except Exception as e:
        print(f"\nAn unexpected error occurred while testing the retriever: {e}")

if __name__ == "__main__":
    # Cek apakah ada topik kustom yang diberikan dari command line
    if len(sys.argv) > 1:
        # Gunakan argumen yang diberikan sebagai topik
        custom_topic = " ".join(sys.argv[1:])
        test_retrieval(custom_topic)
    else:
        # Jika tidak ada argumen, jalankan daftar tes default
        print("\nNo custom topic provided. Running default test suite...")
        default_topics = [
            "dna",
            "reproduction",
            "virus",
            "evolution",
            "photosynthesis",
            "respiration"
        ]
        for topic in default_topics:
            test_retrieval(topic)

    print("\n" + "=" * 60)
    print("✅ Retriever test script finished.")