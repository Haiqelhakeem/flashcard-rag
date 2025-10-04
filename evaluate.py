import os
import time
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Konfigurasi dan Inisialisasi ---
load_dotenv()

# Inisialisasi model LLM yang akan digunakan untuk semua langkah
# Kita bisa menggunakan model yang cepat seperti Flash
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", 
        google_api_key=os.environ.get("GEMINI_API_KEY"),
        temperature=0.0 # Kita ingin jawaban yang deterministik dan faktual
    )
    if not os.environ.get("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY environment variable not found.")
except Exception as e:
    print(f"Error initializing the model: {e}")
    exit()

# --- Template Prompt untuk Setiap Langkah ---

# 1. Prompt untuk Question Generation (QG)
qg_prompt_template = """
Berdasarkan kalimat berikut, buatlah satu pertanyaan yang jawabannya dapat ditemukan di dalam kalimat ini.

Kalimat: "{definition}"

Pertanyaan:
"""
qg_prompt = ChatPromptTemplate.from_template(qg_prompt_template)

# 2. Prompt untuk Question Answering (QA) dari sumber
qa_source_prompt_template = """
Berdasarkan HANYA pada konteks berikut, jawablah pertanyaan di bawah ini. Jika jawaban tidak ada di dalam konteks, tulis "Tidak ditemukan".

Konteks: "{context}"
Pertanyaan: "{question}"

Jawaban:
"""
qa_source_prompt = ChatPromptTemplate.from_template(qa_source_prompt_template)

# 3. Prompt untuk Question Answering (QA) dari ringkasan/definisi
qa_summary_prompt_template = """
Berdasarkan HANYA pada konteks berikut, jawablah pertanyaan di bawah ini.

Konteks: "{definition}"
Pertanyaan: "{question}"

Jawaban:
"""
qa_summary_prompt = ChatPromptTemplate.from_template(qa_summary_prompt_template)


# 4. Prompt untuk LLM as a Judge
judge_prompt_template = """
Bandingkan Jawaban A dan Jawaban B. Apakah keduanya memiliki makna inti yang sama secara faktual? Jawab HANYA dengan kata "SAMA" atau "BEDA".

Jawaban A: "{answer_a}"
Jawaban B: "{answer_b}"

Evaluasi:
"""
judge_prompt = ChatPromptTemplate.from_template(judge_prompt_template)


# --- Chain LangChain untuk Setiap Langkah ---
question_generation_chain = qg_prompt | llm | StrOutputParser()
answer_from_source_chain = qa_source_prompt | llm | StrOutputParser()
answer_from_summary_chain = qa_summary_prompt | llm | StrOutputParser()
judge_chain = judge_prompt | llm | StrOutputParser()


def evaluate_qags_score(flashcard: dict, context: str) -> int:
    """
    Mengevaluasi satu flashcard menggunakan metode QAGS dan mengembalikan skor (1 atau 0).
    """
    definition = flashcard.get("definition", "")
    if not definition or not context:
        return 0 # Tidak bisa dievaluasi jika input kosong

    print("\n" + "="*50)
    print(f"Mengevaluasi Term: {flashcard.get('term', 'N/A')}")
    print(f"Definisi: {definition}")
    print("="*50)

    # Langkah 1: Hasilkan Pertanyaan (QG)
    generated_question = question_generation_chain.invoke({"definition": definition})
    print(f"1. Pertanyaan yang Dihasilkan: {generated_question}")
    time.sleep(5)

    # Langkah 2: Jawab dari Sumber Asli (QA - Source)
    answer_from_source = answer_from_source_chain.invoke({
        "context": context,
        "question": generated_question
    })
    print(f"2. Jawaban dari Sumber Asli: {answer_from_source}")
    time.sleep(5)

    # Langkah 3: Jawab dari Definisi Flashcard (QA - Summary)
    answer_from_summary = answer_from_summary_chain.invoke({
        "definition": definition,
        "question": generated_question
    })
    print(f"3. Jawaban dari Definisi: {answer_from_summary}")
    
    # Cek cepat jika salah satu jawaban tidak ditemukan
    if "tidak ditemukan" in answer_from_source.lower():
        print("4. Evaluasi: Jawaban tidak ditemukan di sumber asli. Skor = 0")
        return 0
    
    time.sleep(5)

    # Langkah 4: Bandingkan Kedua Jawaban (LLM as a Judge)
    evaluation_result = judge_chain.invoke({
        "answer_a": answer_from_source,
        "answer_b": answer_from_summary
    })
    print(f"4. Hasil Evaluasi Juri: {evaluation_result}")
    time.sleep(5)

    # Langkah 5: Tentukan Skor
    if "SAMA" in evaluation_result:
        print(">>> KESIMPULAN: Jawaban SAMA. Skor = 1 (Faktual)")
        return 1
    else:
        print(">>> KESIMPULAN: Jawaban BEDA. Skor = 0 (Tidak Faktual/Halusinasi)")
        return 0

# --- Contoh Penggunaan ---
if __name__ == "__main__":
    # Ini adalah contoh data yang akan Anda dapatkan dari sistem RAG Anda
    sample_flashcard = {
        "term": "Fotosintesis",
        "definition": "Proses biokimia yang mengubah energi cahaya menjadi energi kimia (gula), yang terjadi di dalam kloroplas pada organisme seperti tumbuhan."
    }
    sample_context = """
    Fotosintesis adalah proses vital yang digunakan oleh tumbuhan, alga, dan beberapa bakteri untuk mengubah energi cahaya, biasanya dari matahari, menjadi energi kimia. 
    Proses ini terjadi di dalam organel sel yang disebut kloroplas dan bahan utamanya adalah karbon dioksida dan air. 
    Hasil akhir dari fotosintesis adalah glukosa (gula), yang digunakan oleh organisme sebagai sumber energi, dan oksigen yang dilepaskan ke atmosfer.
    """

    # Jalankan fungsi evaluasi
    final_score = evaluate_qags_score(sample_flashcard, sample_context)
    
    print("\n" + "="*50)
    print(f"Skor Akhir untuk Flashcard Ini: {final_score}")
    print("="*50)