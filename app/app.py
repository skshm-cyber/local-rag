import os
import sys
import time
import subprocess
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

OLLAMA_HOST = "http://127.0.0.1:11434"

def check_ollama():
    try:
        import requests
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=2)
        return resp.status_code == 200
    except:
        return False

OLLAMA_AVAILABLE = check_ollama()

import streamlit as st
from multimodal_rag import (
    get_vector_store,
    get_llm,
    get_vision_llm,
    sync_multimodal_data,
    query_multimodal,
    build_multimodal_prompt,
    analyze_image_with_vlm,
    get_image_from_path,
    harness_execute,
    generate_visualization_code,
    context_collapse_summary,
    RETRIEVAL_K,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MAX_CONTEXT_CHARS,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data")

LLM_MODEL = "phi3"
VISION_MODEL = "llava"

os.makedirs(DATA_PATH, exist_ok=True)

st.set_page_config(page_title="Local AI Chat", page_icon="💬", layout="wide", initial_sidebar_state="expanded")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "context_summarized" not in st.session_state:
    st.session_state.context_summarized = False

st.markdown("""
<style>
    .stChat {
        background: transparent;
    }
    .chat-container {
        background: #1e1e2e;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
    }
    .user-message {
        background: #2d2d4d;
        padding: 15px;
        border-radius: 15px 15px 0 15px;
        margin: 10px 0;
        border-left: 4px solid #7c3aed;
    }
    .assistant-message {
        background: #262639;
        padding: 15px;
        border-radius: 15px 15px 15px 0;
        margin: 10px 0;
        border-left: 4px solid #10b981;
    }
    .sidebar-content {
        background: #1e1e2e;
        padding: 15px;
        border-radius: 10px;
    }
    .stButton > button {
        border-radius: 10px;
    }
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-online { background: #10b981; }
    .status-offline { background: #ef4444; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🤖 Local AI Assistant")
    
    status_class = "status-online" if OLLAMA_AVAILABLE else "status-offline"
    status_text = "Online" if OLLAMA_AVAILABLE else "Offline"
    st.markdown(f"<span class='status-indicator {status_class}'></span><b>Ollama:</b> {status_text}", unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown("### 📚 Knowledge Base")
    
    try:
        vs_check = get_vector_store()
        current_count = vs_check._collection.count()
    except:
        current_count = 0
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Documents", current_count)
    with col2:
        st.metric("Chunk Size", CHUNK_SIZE)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Sync PDFs", use_container_width=True):
            with st.spinner("Processing..."):
                stats = sync_multimodal_data()
                st.success(f"Indexed: {stats['text_chunks']} chunks, {stats['images']} images")
                st.rerun()
    
    with col2:
        if st.button("🗑️ Clear DB", use_container_width=True):
            try:
                vs_check.delete_collection()
            except:
                pass
            st.warning("Database cleared!")
            st.rerun()
    
    st.divider()
    
    st.markdown("### 📁 Files")
    available_files = [f for f in os.listdir(DATA_PATH) if f.lower().endswith(".pdf")]
    if available_files:
        for f in available_files:
            st.text(f"📄 {f}")
    else:
        st.caption("No PDFs in /data folder")
    
    st.divider()
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.context_summarized = False
        st.rerun()
    
    st.markdown("""
    <div style='font-size: 12px; color: #888; margin-top: 20px;'>
        <b>How to use:</b><br>
        1. Add PDFs to /data folder<br>
        2. Click "Sync PDFs" to index<br>
        3. Ask questions below<br>
        <br>
        <b>Fully offline - no internet required</b>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center; color: #7c3aed;'>💬 Chat with Your Documents</h2>", unsafe_allow_html=True)

if not OLLAMA_AVAILABLE:
    st.error("⚠️ Ollama is not running. Please start Ollama to use this app.")
    st.stop()

total_chars = sum(len(m.get("content", "")) for m in st.session_state.messages)
if total_chars > MAX_CONTEXT_CHARS:
    if not st.session_state.context_summarized:
        with st.spinner("Summarizing conversation..."):
            summary = context_collapse_summary(st.session_state.messages)
            st.session_state.messages = [{"role": "system", "content": f"[SUMMARY] {summary}"}]
            st.session_state.context_summarized = True

chat_container = st.container()

with chat_container:
    for msg in st.session_state.messages:
        if msg.get("role") == "system":
            continue
        
        role = msg["role"]
        content = msg["content"]
        
        if role == "user":
            st.markdown(f"""
            <div class='user-message'>
                <b>👤 You:</b><br>
                {content}
            </div>
            """, unsafe_allow_html=True)
        else:
            sources = msg.get("sources", [])
            has_images = any(s.get("type") == "image" for s in sources)
            
            st.markdown(f"""
            <div class='assistant-message'>
                <b>🤖 Assistant:</b><br>
                {content}
            </div>
            """, unsafe_allow_html=True)
            
            if has_images:
                img_sources = [s for s in sources if s.get("type") == "image"]
                cols = st.columns(min(len(img_sources), 2))
                for idx, s in enumerate(img_sources):
                    img_path = s.get("image_path")
                    if img_path and os.path.exists(img_path):
                        with cols[idx % 2]:
                            st.image(img_path, caption=f"Page {s.get('page', '?')}", use_container_width=True)
            
            if sources:
                text_sources = [s for s in sources if s.get("type") != "image"]
                if text_sources:
                    with st.expander(f"📚 Sources ({len(text_sources)})"):
                        for s in text_sources[:5]:
                            st.markdown(f"**Page {s.get('page', 0)}**: {s.get('text', '')[:150]}...")

st.divider()

col1, col2 = st.columns([5, 1])
with col1:
    prompt = st.chat_input("Type your message here...")
with col2:
    visualize = st.toggle("📊 Viz", value=False, help="Generate chart from data")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    
    with st.spinner("Thinking..."):
        docs, dists = query_multimodal(prompt)
        
        if visualize and docs:
            with st.spinner("Generating visualization..."):
                viz_code = generate_visualization_code(docs)
                if viz_code != "NO_DATA":
                    st.code(viz_code, language="python")
                else:
                    st.info("No data for visualization")
        
        answer = harness_execute(prompt, docs, dists)
        
        sources = []
        if docs:
            for d, dist in zip(docs, dists):
                content_type = d.metadata.get("content_type", "text")
                source = {
                    "file": d.metadata.get("source_file", "Unknown"),
                    "page": d.metadata.get("page", 0),
                    "dist": dist,
                    "text": d.page_content[:200],
                    "type": content_type,
                }
                if content_type == "image":
                    source["image_path"] = d.metadata.get("image_path")
                    source["image_id"] = d.metadata.get("image_id")
                sources.append(source)
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer, 
            "sources": sources
        })
    
    st.rerun()
