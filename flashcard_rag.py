import os
import json
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langsmith import traceable
from utils import get_vector_db_retriever

load_dotenv()

MODEL_NAME = "gemini-2.0-flash"

try:
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=os.environ.get("GEMINI_API_KEY")
    )
    if not os.environ.get("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY environment variable not found.")
except Exception as e:
    print(f"Error initializing the model: {e}")
    exit()

FLASHCARD_PROMPT = """
Tugas Anda adalah menjadi seorang guru ahli yang membuat materi pembelajaran yang mudah dihafal. Berdasarkan konteks yang diberikan, buatlah satu set flashcard dalam **Bahasa Indonesia** yang berfokus pada topik utama: **{topic}**.

Setiap flashcard harus mengikuti prinsip-prinsip berikut agar mudah dihafal:
1.  **Satu Konsep Utama**: Setiap flashcard harus fokus pada satu ide atau istilah kunci saja.
2.  **Bahasa Sederhana**: Gunakan bahasa yang sederhana dan mudah dimengerti, bukan jargon akademis yang rumit.
---
## Contoh 1
### Topik: Fotosintesis
### Konteks:
Mitokondria adalah organel sel yang dikenal sebagai "pembangkit tenaga" sel. Fungsi utamanya adalah menghasilkan adenosin trifosfat (ATP). Fotosintesis adalah proses yang digunakan oleh tumbuhan untuk mengubah energi cahaya menjadi energi kimia, yang terjadi di dalam kloroplas.
### Output JSON:
[
  {{
    "term": "Fotosintesis",
    "definition": "Proses biokimia yang mengubah energi cahaya menjadi energi kimia (gula), yang terjadi di dalam kloroplas pada organisme seperti tumbuhan."
  }}
]
---
## Contoh 2
### Topik: Struktur DNA
### Konteks:
Asam deoksiribonukleat, atau DNA, adalah molekul yang membawa instruksi genetik. RNA juga merupakan molekul penting. DNA terdiri dari dua untai yang melingkar membentuk heliks ganda.
### Output JSON:
[
  {{
    "term": "Struktur DNA",
    "definition": "Molekul berbentuk heliks ganda yang terdiri dari dua untai polinukleotida dan membawa instruksi genetik."
  }}
]
---
## Tugas Anda
### Topik: {topic}
### Konteks:
{context}

Jika konteks di atas kosong atau tidak relevan, jangan menebak atau menggunakan pengetahuan umum. 
Kembalikan output berikut:
[]

### Output JSON:
"""

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

@traceable(run_type="chain", name="RAG_Flashcard_Chain_With_Topic")
def generate_flashcard_data(topic: str):
    retriever = get_vector_db_retriever()
    parser = JsonOutputParser()
    prompt = ChatPromptTemplate.from_template(template=FLASHCARD_PROMPT)
    generation_chain = prompt | llm | parser

    def safe_generate(inputs):
        context_docs = inputs["context"]
        formatted_context = format_docs(context_docs)
        if not formatted_context.strip():
            return []  # ⛔️ Prevent hallucination
        return generation_chain.invoke({
            "context": formatted_context,
            "topic": inputs["topic"]
        })

    chain = (
        RunnablePassthrough.assign(
            context=lambda inputs: retriever.invoke(inputs["topic"])
        )
        .assign(
            flashcards=RunnableLambda(safe_generate)
        )
    )

    try:
        result = chain.invoke({"topic": topic})
        return result
    except Exception as e:
        print(f"An error occurred in the chain: {e}")
        return None
