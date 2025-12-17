import os
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langsmith import traceable
from utils import get_vector_db_retriever

load_dotenv()

# FIX 1: Use 1.5-flash for stability and free tier limits
MODEL_NAME = "gemini-2.5-flash"

try:
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=os.environ.get("GEMINI_API_KEY"),
        temperature=0.4
    )
    if not os.environ.get("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY environment variable not found.")
except Exception as e:
    print(f"Error initializing the model: {e}")
    exit()

# FIX 2: Strict Prompt with Double Braces {{ }} for JSON examples
FLASHCARD_PROMPT = """
Anda adalah asisten dosen biologi ahli. Tugas Anda adalah mengekstrak istilah penting dan definisinya dari konteks yang diberikan menjadi format JSON.

**Aturan Sangat Penting (JANGAN DILANGGAR):**
1. Output WAJIB berupa **JSON List** valid.
2. Key (kunci) JSON harus persis **"term"** dan **"definition"**. 
3. **JANGAN terjemahkan** kata "term" atau "definition" ke bahasa Indonesia. Tetap gunakan bahasa Inggris untuk key tersebut.
4. Isi (value) dari "term" dan "definition" harus dalam **Bahasa Indonesia**.
5. Buatlah antara 10 sampai 20 item.

### Contoh Format Output yang BENAR:
[
  {{
    "term": "Fotosintesis",
    "definition": "Proses tumbuhan mengubah cahaya menjadi energi kimia."
  }},
  {{
    "term": "Mitokondria",
    "definition": "Organel sel yang berfungsi sebagai tempat respirasi sel."
  }}
]

### Tugas Anda
Topik: {topic}
Konteks:
{context}

Jika konteks di atas kosong atau tidak relevan, jangan menebak. Kembalikan output berikut:
[]

### Output JSON:
"""

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

@traceable(run_type="chain", name="RAG_Flashcard_Chain_With_Topic")
def generate_flashcard_data(topic: str):
    # 1. Access VectorStore
    retriever = get_vector_db_retriever()
    vectorstore = retriever.vectorstore 
    
    # 2. Retrieve (Fast k=8)
    try:
        results_with_scores = vectorstore.similarity_search_with_score(topic, k=10)
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        return None

    # FIX 3: Distance Threshold Logic
    MAX_DISTANCE_THRESHOLD = 25.0 
    
    relevant_docs = []
    print(f"\n--- 🔍 Filtering Context for '{topic}' ---")
    for doc, distance_score in results_with_scores:
        if distance_score <= MAX_DISTANCE_THRESHOLD:
            # FIX 4: Safety Truncate (Prevents massive tokens)
            # We truncate the content for the LLM, but we keep the object alive
            if len(doc.page_content) > 1500:
                doc.page_content = doc.page_content[:1500] + "...(truncated)"
            
            relevant_docs.append(doc)
            print(f"✅ ACCEPTED (Dist: {distance_score:.2f})")
        else:
            print(f"❌ REJECTED (Dist: {distance_score:.2f})")

    # 5. Handle Empty Results
    if not relevant_docs:
        print("⚠️ All documents were rejected by the threshold.")
        return None

    # 6. Generate Flashcards
    formatted_context = format_docs(relevant_docs)
    
    parser = JsonOutputParser()
    prompt = ChatPromptTemplate.from_template(template=FLASHCARD_PROMPT)
    generation_chain = prompt | llm | parser

    try:
        flashcards = generation_chain.invoke({
            "context": formatted_context,
            "topic": topic
        })
        
        # FIX 5: CRITICAL - Ensure 'context' is returned so App can display it
        return {
            "flashcards": flashcards,
            "context": relevant_docs 
        }
    except Exception as e:
        print(f"Error in generation chain: {e}")
        return None