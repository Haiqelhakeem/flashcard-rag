import pandas as pd
import json
from flashcard_rag import generate_flashcard_data # <-- Impor fungsi Anda

# Daftar topik yang ingin Anda proses
DAFTAR_TOPIK = [
    "Bagian tumbuhan",
    "DNA",
    "Fotosintesis",
    "Photosynthesis",
    "Evolution",
    "Virus",
    "Symbiosis",
    "Reproduction",
    "Struktur Heliks Ganda DNA",
    "Sistem Pencernaan",
    "Hubungan Organisme"
]

def buat_file_output_rag():
    """
    Menjalankan RAG untuk setiap topik dan menyimpan hasilnya 
    (termasuk contexts) ke dalam file CSV.
    """
    print("🚀 Memulai proses pembuatan output RAG...")
    
    hasil_semua_topik = []

    for topik in DAFTAR_TOPIK:
        print(f"🔄 Memproses topik: '{topik}'...")
        
        try:
            # 1. Panggil fungsi Anda yang sudah ada
            hasil_rag = generate_flashcard_data(topik)

            if not hasil_rag:
                print(f"⚠️ Gagal memproses topik '{topik}'.")
                continue

            # 2. Ekstrak data dari dictionary hasil
            flashcards = hasil_rag.get("flashcards", [])
            dokumen_konteks = hasil_rag.get("context", [])
            
            # 3. Ubah list of Document menjadi list of strings (ini penting!)
            list_string_konteks = [doc.page_content for doc in dokumen_konteks]

            # 4. Simpan semua data yang diperlukan
            hasil_semua_topik.append({
                "question": topik,
                "rag_output": json.dumps(flashcards, ensure_ascii=False), # ensure_ascii untuk karakter non-latin
                "contexts": str(list_string_konteks) # Simpan sebagai string dari list
            })
            print(f"✅ Berhasil.")

        except Exception as e:
            print(f"❌ Terjadi error pada topik '{topik}': {e}")

    # 5. Simpan semua hasil ke dalam satu file CSV
    if hasil_semua_topik:
        df_hasil = pd.DataFrame(hasil_semua_topik)
        df_hasil.to_csv("rag_output_dataset.csv", index=False, sep=';')
        print("\n\n✅ File 'rag_output_dataset.csv' berhasil dibuat dengan kolom contexts!")
    else:
        print("\n\n❌ Tidak ada hasil yang bisa disimpan.")


if __name__ == "__main__":
    buat_file_output_rag()