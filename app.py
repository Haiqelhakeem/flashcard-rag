import streamlit as st
import asyncio
try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
import os
import time
from io import BytesIO
from flashcard_rag import generate_flashcard_data

# --- PDF Generation Libraries ---
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="RAG Flashcards Generator", layout="wide")
st.title("Biology Flashcards Generator - RAG")
st.write("Masukkan keyword Biologi dan AI akan membuatkan flashcards dari keyword yang dimasukkan berdasarkan dokumen yang relevan")

# Exact colors from your UI
CARD_COLORS = [
    "#74c0fc",  # Bright Blue
    "#ff8787",  # Strong Pink/Red
    "#ffd43b",  # Vibrant Yellow
    "#69db7c",  # Bright Green
    "#ffa94d",  # Orange
    "#d0bfff",  # Lilac
    "#66d9e8",  # Cyan
]

# --- Initialize Session State ---
# This "Memory" keeps the data alive when buttons are clicked
if "flashcards" not in st.session_state:
    st.session_state.flashcards = None
if "source_docs" not in st.session_state:
    st.session_state.source_docs = None
if "topic" not in st.session_state:
    st.session_state.topic = ""
if "duration" not in st.session_state:
    st.session_state.duration = 0

def create_pdf(flashcards_data, topic):
    """
    Generates a PDF with consistent rectangular boxes.
    Left side = Colored (Term). Right side = White (Definition).
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    
    elements = []
    styles = getSampleStyleSheet()
    
    # --- Styles ---
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], alignment=TA_CENTER, fontSize=20, spaceAfter=10)
    subtitle_style = ParagraphStyle('Subtitle', parent=styles['Normal'], alignment=TA_CENTER, fontSize=10, textColor=colors.grey, spaceAfter=20)
    term_style = ParagraphStyle('TermStyle', parent=styles['Normal'], alignment=TA_CENTER, fontSize=14, leading=18, fontName='Helvetica-Bold', textColor=colors.black)
    def_style = ParagraphStyle('DefStyle', parent=styles['Normal'], alignment=TA_LEFT, fontSize=11, leading=15, textColor=colors.black)
    header_style = ParagraphStyle('HeaderStyle', parent=styles['Normal'], alignment=TA_CENTER, fontSize=12, fontName='Helvetica-Bold')

    # --- Header Content ---
    elements.append(Paragraph(f"Flashcards: {topic}", title_style))
    elements.append(Paragraph("(Print, Cut horizontally, then Fold in the middle)", subtitle_style))

    # --- Table Data Preparation ---
    data = [[Paragraph("TERM (Front)", header_style), Paragraph("DEFINITION (Back)", header_style)]]
    
    row_colors = []
    for i, card in enumerate(flashcards_data):
        term = card.get("term") or "No term"
        definition = card.get("definition") or "No definition"
        data.append([Paragraph(term, term_style), Paragraph(definition, def_style)])
        row_colors.append(CARD_COLORS[i % len(CARD_COLORS)])

    # --- Table Dimensions ---
    usable_width = A4[0] - 60
    col_width = usable_width / 2
    
    # === CONSISTENT HEIGHT LOGIC ===
    FIXED_CARD_HEIGHT = 150 
    row_heights = [30] + [FIXED_CARD_HEIGHT] * len(flashcards_data)

    t = Table(data, colWidths=[col_width, col_width], rowHeights=row_heights)
    
    # --- Table Styling ---
    style_cmds = [
        ('GRID', (0, 0), (-1, -1), 1, colors.black),   
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),        
        ('ALIGN', (0, 0), (0, -1), 'CENTER'),          
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey) # Header
    ]

    for i, hex_color in enumerate(row_colors):
        row_idx = i + 1
        bg_color = colors.HexColor(hex_color)
        style_cmds.append(('BACKGROUND', (0, row_idx), (0, row_idx), bg_color))

    t.setStyle(TableStyle(style_cmds))
    elements.append(t)
    doc.build(elements)
    buffer.seek(0)
    return buffer

# --- User Input ---
user_topic = st.text_input("Masukkan Keyword:", placeholder="contoh: DNA")

# When "Buat" is clicked, we ONLY update the Session State
if st.button("Buat Flashcards", type="primary"):
    if user_topic:
        with st.spinner("Menganalisis dokumen dan membuat flashcard..."):
            start_time = time.time()
            result = generate_flashcard_data(user_topic)
            end_time = time.time()
            
            # SAVE TO SESSION STATE
            if result:
                st.session_state.flashcards = result.get("flashcards")
                st.session_state.source_docs = result.get("context")
                st.session_state.topic = user_topic
                st.session_state.duration = end_time - start_time
            else:
                st.error("Gagal membuat flashcards (tidak ada hasil).")
    else:
        st.warning("Please enter a topic first.")

# --- Display Logic (Runs if data exists in memory) ---
if st.session_state.flashcards:
    flashcards_data = st.session_state.flashcards
    source_documents = st.session_state.source_docs
    topic = st.session_state.topic
    duration = st.session_state.duration

    st.success(f"Berhasil membuat {len(flashcards_data)} flashcards mengenai topik '{topic}'.")
    st.write(f"Durasi: {duration:.2f} detik")

    # --- 1. Generate PDF Button ---
    pdf_file = create_pdf(flashcards_data, topic)
    st.download_button(
        label="📄 Download Printable Flashcards (PDF)",
        data=pdf_file,
        file_name=f"Flashcards_{topic}.pdf",
        mime="application/pdf",
        use_container_width=True
    )

    # --- 2. Display Flashcards (UI) ---
    cols = st.columns(3)
    for i, card in enumerate(flashcards_data):
        col = cols[i % 3]
        term = card.get("term") or "No term"
        definition = card.get("definition") or "No definition"
        current_color = CARD_COLORS[i % len(CARD_COLORS)]
        
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
    
    # --- 3. Source Docs (Updated Section) ---
    st.markdown("---")
    st.subheader("Sumber yang digunakan untuk membuat flashcard tersebut:")
        
    if source_documents:
        for doc in source_documents:
            # Safely get filename, default to "Unknown" if missing
            source_file = os.path.basename(doc.metadata.get("source", "Unknown File"))
            # Safely get page number, default to 0 if missing (display as 1)
            page_number = doc.metadata.get("page", -1) + 1 
                
            with st.expander(f"**Source:** {source_file}, **Page:** {page_number}"):
                st.write(doc.page_content)