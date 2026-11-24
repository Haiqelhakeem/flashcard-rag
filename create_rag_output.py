import pandas as pd
import json
import time
from flashcard_rag import generate_flashcard_data 

# Daftar topik (Total 11 item)
DAFTAR_TOPIK = [
    # "Keanekaragaman Hayati dan Klasifikasi",
    # "Virus",
    # "Monera",
    # "Protista",
    # "Fungi",
    # "Metabolisme",
    # "Pembelahan Sel",
    # "Substansi Genetika",
    # "Genetika Manusia",
    # "Mutasi"
    # "Bagian tumbuhan",
    # "DNA",
    # "Fotosintesis",
    # "Photosynthesis",
    # "Evolution",
    # "Virus",
    "Symbiosis",
    "Reproduction",
    "Struktur Heliks Ganda DNA",
    "Hubungan Organisme"
]

def buat_file_output_rag():
    print("🚀 Memulai proses pembuatan output RAG untuk Evaluasi...")
    
    hasil_semua_topik = []

    for i, topik in enumerate(DAFTAR_TOPIK):
        print(f"[{i+1}/{len(DAFTAR_TOPIK)}] 🔄 Memproses topik: '{topik}'...")
        
        try:
            # Panggil fungsi RAG
            hasil_rag = generate_flashcard_data(topik)

            if not hasil_rag:
                print(f"⚠️ Warning: Topik '{topik}' tidak menghasilkan output (mungkin terfilter).")
                # Tetap simpan baris kosong agar kita tahu topik mana yang gagal
                hasil_semua_topik.append({
                    "question": topik,
                    "rag_output": "FAILED / FILTERED",
                    "contexts": "[]"
                })
                continue

            # Ekstrak data
            flashcards = hasil_rag.get("flashcards", [])
            dokumen_konteks = hasil_rag.get("context", [])
            
            # Konversi objek Dokumen ke List String untuk CSV
            # Kita ambil page_content saja
            list_string_konteks = [doc.page_content for doc in dokumen_konteks]

            # Simpan
            hasil_semua_topik.append({
                "question": topik,
                "rag_output": json.dumps(flashcards, ensure_ascii=False),
                "contexts": json.dumps(list_string_konteks, ensure_ascii=False) # Pakai JSON dumps agar rapi di CSV
            })
            print(f"✅ Berhasil ({len(flashcards)} flashcards).")

        except Exception as e:
            print(f"❌ Error pada topik '{topik}': {e}")
            hasil_semua_topik.append({
                "question": topik,
                "rag_output": f"ERROR: {e}",
                "contexts": "[]"
            })

        # --- SAFETY PAUSE ---
        # Berhenti 10 detik setiap selesai satu topik untuk menghindari Limit Kuota 429
        # Gemini Free Tier punya limit Request Per Minute (RPM)
        print("⏳ Menunggu 10 detik agar kuota aman...")
        time.sleep(10) 

    # Simpan ke CSV
    if hasil_semua_topik:
        # Gunakan separator semicolon (;) agar aman dibuka di Excel Indonesia
        filename = "rag_output_dataset_symbiosis_hubunganOrganisme.csv"
        df_hasil = pd.DataFrame(hasil_semua_topik)
        df_hasil.to_csv(filename, index=False, sep=';')
        print(f"\n\n🎉 Selesai! File '{filename}' berhasil dibuat.")
    else:
        print("\n\n❌ Tidak ada hasil yang bisa disimpan.")

if __name__ == "__main__":
    buat_file_output_rag()