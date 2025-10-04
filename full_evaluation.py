import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import json
import time

# Impor fungsi-fungsi dari file lain di proyek Anda
from flashcard_rag import generate_flashcard_data, format_docs
from evaluate import evaluate_qags_score
from utils import get_vector_db_retriever

# --- Konfigurasi dan Inisialisasi ---
load_dotenv()

# Daftar topik yang akan digunakan untuk mengevaluasi kedua sistem
EVALUATION_TOPICS = [
    "dna",
    "photosynthesis",
    "respiration",
    "virus",
    "reproduction",
    "cell structure",
    "mitosis",
    "genetics",
    "ecosystem",
    "human anatomy"
]

# Inisialisasi model LLM untuk sistem LLM-Only
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.environ.get("GEMINI_API_KEY"),
        temperature=0.0
    )
    if not os.environ.get("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY environment variable not found.")
except Exception as e:
    print(f"Error initializing the model: {e}")
    exit()

# --- Fungsi untuk Sistem LLM-Only ---

# Prompt untuk menghasilkan flashcard TANPA konteks (hanya berdasarkan pengetahuan umum LLM)
llm_only_prompt_template = """
Anda adalah seorang guru ahli. Buatlah satu flashcard dalam Bahasa Indonesia mengenai topik: **{topic}**.
Flashcard harus memiliki "term" dan "definition". Gunakan bahasa yang sederhana dan mudah dimengerti.

Format output Anda harus berupa JSON tunggal seperti ini:
{{
  "term": "Istilah Kunci",
  "definition": "Definisi yang jelas dan ringkas."
}}

Topik: {topic}
Output JSON:
"""
llm_only_prompt = ChatPromptTemplate.from_template(llm_only_prompt_template)
# Parser untuk memastikan outputnya adalah JSON yang valid
# Menggunakan JsonOutputParser sederhana karena kita mengharapkan satu objek, bukan list
llm_only_parser = JsonOutputParser() 
llm_only_chain = llm_only_prompt | llm | llm_only_parser

def generate_llm_only_flashcard(topic: str) -> dict:
    """
    Menghasilkan satu flashcard menggunakan LLM saja, tanpa RAG.
    """
    try:
        flashcard = llm_only_chain.invoke({"topic": topic})
        return flashcard
    except Exception as e:
        print(f"  - Error saat membuat flashcard LLM-Only untuk '{topic}': {e}")
        return None

# --- Fungsi Utama Evaluasi ---

def run_evaluation():
    """
    Menjalankan proses evaluasi untuk sistem RAG dan LLM-Only
    dan membandingkan skor faithfulness mereka.
    """
    print("=" * 70)
    print("🚀 Memulai Proses Evaluasi Faithfulness Model")
    print("=" * 70)

    rag_scores = []
    llm_only_scores = []
    
    # Dapatkan retriever sekali di awal agar efisien
    try:
        retriever = get_vector_db_retriever()
    except Exception as e:
        print(f"CRITICAL ERROR: Tidak dapat memuat retriever. Proses evaluasi dihentikan. Error: {e}")
        return

    for i, topic in enumerate(EVALUATION_TOPICS):
        print(f"\n--- Mengevaluasi Topik {i+1}/{len(EVALUATION_TOPICS)}: '{topic}' ---")

        # --- 1. Evaluasi Sistem RAG ---
        print("\n  [Sistem RAG]")
        rag_result = generate_flashcard_data(topic)
        if rag_result and rag_result.get("flashcards"):
            # Kita hanya evaluasi flashcard pertama yang dihasilkan untuk konsistensi
            rag_flashcard = rag_result["flashcards"][0]
            # Konteks didapat langsung dari hasil RAG
            rag_context_docs = rag_result.get("context", [])
            rag_context_text = format_docs(rag_context_docs)
            
            # Panggil fungsi evaluasi QAGS
            rag_score = evaluate_qags_score(rag_flashcard, rag_context_text)
            rag_scores.append(rag_score)
            print(f"  - Skor Faithfulness RAG: {rag_score}")
        else:
            print("  - Gagal menghasilkan flashcard dari sistem RAG.")
            rag_scores.append(0) # Anggap skor 0 jika gagal

        # --- 2. Evaluasi Sistem LLM-Only ---
        print("\n  [Sistem LLM-Only]")
        llm_only_flashcard = generate_llm_only_flashcard(topic)
        if llm_only_flashcard:
            # PENTING: Untuk perbandingan yang adil, kita evaluasi flashcard LLM-Only
            # terhadap konteks "ground truth" yang SEHARUSNYA digunakan.
            print("  - Mengambil konteks ground truth untuk perbandingan...")
            ground_truth_docs = retriever.invoke(topic)
            ground_truth_text = format_docs(ground_truth_docs)

            if not ground_truth_text:
                print("  - Tidak ditemukan konteks ground truth, evaluasi untuk topik ini dilewati.")
                continue

            # Panggil fungsi evaluasi QAGS
            llm_only_score = evaluate_qags_score(llm_only_flashcard, ground_truth_text)
            llm_only_scores.append(llm_only_score)
            print(f"  - Skor Faithfulness LLM-Only: {llm_only_score}")
        else:
            print("  - Gagal menghasilkan flashcard dari sistem LLM-Only.")
            llm_only_scores.append(0) # Anggap skor 0 jika gagal
        
    # --- 3. Tampilkan Hasil Akhir ---
    print("\n" + "=" * 70)
    print("📊 Hasil Akhir Evaluasi")
    print("=" * 70)

    # Hitung skor rata-rata
    avg_rag_score = (sum(rag_scores) / len(rag_scores)) * 100 if rag_scores else 0
    avg_llm_only_score = (sum(llm_only_scores) / len(llm_only_scores)) * 100 if llm_only_scores else 0
    
    print(f"Jumlah Topik yang Dievaluasi : {len(EVALUATION_TOPICS)}")
    print(f"Skor Faithfulness Sistem RAG      : {avg_rag_score:.2f}%")
    print(f"Skor Faithfulness Sistem LLM-Only : {avg_llm_only_score:.2f}%")
    print("-" * 70)
    
    if avg_rag_score > avg_llm_only_score:
        print("Kesimpulan: Sistem RAG secara signifikan lebih faktual dan mengurangi halusinasi dibandingkan LLM-Only.")
    else:
        print("Kesimpulan: Sistem RAG tidak menunjukkan peningkatan faithfulness yang signifikan dibandingkan LLM-Only.")


if __name__ == "__main__":
    run_evaluation()