import streamlit as st
import os
import tempfile
from pathlib import Path
from src.rag_engine import RAGEngine

# Page configuration and aesthetic styles
st.set_page_config(page_title="Axolotl RAG", page_icon="🦎", layout="wide")

# Premium CSS styles (Glassmorphism and dark mode)
st.markdown("""
<style>
    /* Modern typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    
    /* Custom sidebar */
    [data-testid="stSidebar"] {
        background: rgba(20, 20, 25, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Chat container */
    .stChatMessage {
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    /* Gradient titles */
    h1 {
        background: linear-gradient(90deg, #4F46E5, #9333EA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    /* Subtitles */
    .subtitle {
        color: #888;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_engine():
    # Instantiate the engine only once
    return RAGEngine()

try:
    engine = get_engine()
except ValueError as e:
    st.error(f"Configuration error: {e}. Make sure you have your GEMINI_API_KEY in .env")
    st.stop()

# --- Main Interface ---
st.title("🦎 Axolotl RAG")
st.markdown("<div class='subtitle'>Smart search over your documents using Google Gemini</div>", unsafe_allow_html=True)

# --- Sidebar: Document Management ---
with st.sidebar:
    st.header("📚 Documents")
    
    uploaded_files = st.file_uploader("Upload your documents (PDF, TXT, MD)", accept_multiple_files=True)
    
    if st.button("Index Documents", type="primary"):
        if not uploaded_files:
            st.warning("Upload at least one file first.")
        else:
            with st.spinner("Processing documents..."):
                total_chunks = 0
                for uploaded_file in uploaded_files:
                    # Save file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    
                    try:
                        chunks = engine.ingest_file(tmp_path)
                        total_chunks += chunks
                    finally:
                        os.unlink(tmp_path)
                
                st.success(f"Done! Indexed {total_chunks} chunks.")
    
    st.divider()
    stats = engine.vector_store.count()
    st.metric(label="Chunks in Database", value=stats)
    
    if st.button("Clear Database"):
        engine.vector_store.clear()
        st.success("Database cleared. Reload the page.")
        st.rerun()

# --- Chat Area ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("View sources"):
                for idx, src in enumerate(message["sources"]):
                    st.markdown(f"**Source {idx+1}:** `{src.metadata.get('source', 'N/A')}` (Distance: {src.metadata.get('distance', 0.0):.4f})")
                    st.caption(f"{src.text[:200]}...")

# User input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Use query_stream
                stream, sources = engine.query_stream(prompt)
                response = st.write_stream((chunk for chunk in stream if chunk))
                
                if sources:
                    with st.expander("Ver Fuentes Consultadas"):
                        for i, (doc, score) in enumerate(sources):
                            source_name = os.path.basename(doc.metadata.get("source", "unknown"))
                            st.markdown(f"**Fuente {i+1}:** `{source_name}` (Distancia: {score:.4f})")
                            st.caption(f"{doc.page_content[:200]}...")
                            
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "sources": sources
                })
            except Exception as e:
                st.error(f"An error occurred: {e}")
