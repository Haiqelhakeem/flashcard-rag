import streamlit as st
import asyncio
try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
import os
import time
from flashcard_rag import generate_flashcard_data

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="RAG Flashcards Generator", layout="wide")
st.title("Biology Flashcards Generator - RAG")
st.write("Masukkan topik Biologi dan AI akan membuatkan flashcards dari topik yang dimasukkan berdasarkan dokumen yang relevan")

card_colors = [
    "#74c0fc",  # Bright Blue
    "#ff8787",  # Strong Pink/Red
    "#ffd43b",  # Vibrant Yellow
    "#69db7c",  # Bright Green
    "#ffa94d",  # Orange
    "#d0bfff",  # Lilac
    "#66d9e8",  # Cyan
]

# --- User Input ---
user_topic = st.text_input("Masukkan Topik:", placeholder="contoh: DNA")

if st.button("Generate Flashcards!", type="primary"):
    if user_topic:
        with st.spinner("Menganalisis dokumen dan membuat flashcard..."):
            start_time = time.time()
            # The backend function now returns a dictionary with sources
            result = generate_flashcard_data(user_topic)

            end_time = time.time()
            duration = end_time - start_time

        # Extract the flashcards and source documents from the result
        flashcards_data = result.get("flashcards") if result else None
        source_documents = result.get("context") if result else None

        if flashcards_data and isinstance(flashcards_data, list):
            st.success(f"Berhasil membuat {len(flashcards_data)} flashcards mengenai topik '{user_topic}'. Silahkan lihat di bawah!")
            st.write(f"Durasi: {duration:.2f} detik")

            # Display Flashcards (No changes to this part)
            cols = st.columns(3)
            for i, card in enumerate(flashcards_data):
                col = cols[i % 3]
                term = card.get("term", "No term")
                definition = card.get("definition", "No definition")
                current_color = card_colors[i % len(card_colors)]
                component_html = f"""
                <style>
                    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');
                    .card-container-{i} {{ perspective: 1000px; }}
                    .card-{i} {{ position: relative; width: 100%; height: 200px; transition: transform 0.8s; transform-style: preserve-3d; cursor: pointer; border: 1px solid #e6e6e6; border-radius: 12px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
                    .is-flipped-{i} {{ transform: rotateY(180deg); }}
                    .card-face-{i} {{ position: absolute; width: 100%; height: 100%; -webkit-backface-visibility: hidden; backface-visibility: hidden; display: flex; justify-content: center; align-items: center; padding: 20px; box-sizing: border-box; border-radius: 12px; text-align: center; font-family: 'Poppins', sans-serif; font-size: 1.1rem; }}
                    .card-front-{i} {{ background-color: {current_color}; color: #212529; font-weight: 600; }}
                    .card-back-{i} {{ background-color: #f0f2f6; color: #212529; transform: rotateY(180deg); font-weight: 400; }}
                </style>
                <div class="card-container-{i}"><div class="card-{i}" onclick="this.classList.toggle('is-flipped-{i}')"><div class="card-face-{i} card-front-{i}"><p>{term}</p></div><div class="card-face-{i} card-back-{i}"><p>{definition}</p></div></div></div>
                """
                with col:
                    st.components.v1.html(component_html, height=220)

            # --- NEW: Display the sources ---
            st.markdown("---")
            st.subheader("Sumber yang digunakan untuk membuat flashcard tersebut:")
            
            if source_documents:
                for doc in source_documents:
                    source_file = os.path.basename(doc.metadata.get("source", "Unknown File"))
                    page_number = doc.metadata.get("page", -1) + 1 # Add 1 for display
                    
                    with st.expander(f"**Source:** {source_file}, **Page:** {page_number}"):
                        st.write(doc.page_content)
            else:
                st.write("No source documents were retrieved.")
        else:
            st.error("Failed to generate flashcard data. The topic might not be in the documents, or an error occurred.")
    else:
        st.warning("Please enter a topic first.")