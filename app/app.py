import os
import sys
import time
import subprocess
import requests
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

OLLAMA_HOST = "http://127.0.0.1:11434"
OLLAMA_AVAILABLE = False
OLLAMA_MODELS = []

def ensure_ollama_started():
    global OLLAMA_AVAILABLE
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=1)
        if resp.status_code == 200:
            OLLAMA_AVAILABLE = True
            return True
    except:
        pass
    
    ollama_paths = [
        str(project_root / "Ollama.app" / "Contents" / "Resources" / "ollama"),
        "/Applications/Ollama.app/Contents/Resources/ollama"
    ]
    for ollama_path in ollama_paths:
        if os.path.exists(ollama_path):
            subprocess.Popen([ollama_path, "serve"], 
                          stdout=subprocess.DEVNULL, 
                          stderr=subprocess.DEVNULL)
            for _ in range(10):
                time.sleep(1)
                try:
                    resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=2)
                    if resp.status_code == 200:
                        OLLAMA_AVAILABLE = True
                        return True
                except:
                    pass
            break
    return False

ensure_ollama_started()

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

st.set_page_config(page_title="Local RAG: Claude Code Patterns", page_icon="🕵️", layout="wide")
st.title("Local RAG: Three-Layer Intelligence Engine 📚")

st.markdown("""
<div style='background: linear-gradient(90deg, #1a1a2e 0%, #16213e 100%); padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
    <h4 style='color: #00d4ff; margin: 0;'>🔧 Architecture: Claude Code Patterns</h4>
    <p style='color: #aaa; margin: 5px 0 0 0; font-size: 12px;'>
        Ingestion: Contextual Retrieval → Core: The Harness (Plan-Route-Act-Verify) → Viz: Data Analyst
    </p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("🗄️ Knowledge Source")
    
    vs_check = get_vector_store()
    try:
        current_count = vs_check._collection.count()
    except:
        current_count = 0
    st.metric("Total Indexed Chunks", current_count)
    
    st.markdown("**Retrieval Parameters:**")
    st.code(f"Chunk Size: {CHUNK_SIZE} tokens\nOverlap: {CHUNK_OVERLAP}%\nk: {RETRIEVAL_K}", language="text")
    
    st.markdown("Place your PDFs in the `/data` folder.")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Sync Folder", use_container_width=True):
            with st.spinner("Processing PDFs with Contextual Retrieval..."):
                stats = sync_multimodal_data()
                if stats["text_chunks"] > 0 or stats["images"] > 0:
                    st.success(f"Indexed: {stats['text_chunks']} text chunks, {stats['images']} images")
                    st.rerun()
                else:
                    st.info("No new PDFs to process.")
    
    with col2:
        if st.button("🗑️ Clear DB", use_container_width=True):
            vs_check.delete_collection()
            st.warning("Database cleared.")
            st.rerun()

    st.divider()
    st.subheader("Files in Source Folder")
    available_files = [f for f in os.listdir(DATA_PATH) if f.lower().endswith(".pdf")]
    if available_files:
        for f in available_files:
            st.text(f"📄 {f}")
    else:
        st.caption("No PDFs found in /data")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "context_summarized" not in st.session_state:
    st.session_state.context_summarized = False

total_chars = sum(len(m.get("content", "")) for m in st.session_state.messages)
if total_chars > MAX_CONTEXT_CHARS:
    if not st.session_state.context_summarized:
        with st.spinner("Context collapse triggered..."):
            summary = context_collapse_summary(st.session_state.messages)
            st.session_state.messages = [{"role": "system", "content": f"[CONTEXT COLLAPSE] {summary}"}]
            st.session_state.context_summarized = True

for msg in st.session_state.messages:
    if msg.get("role") == "system":
        continue
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        msg_images = [s for s in msg.get("sources", []) if s.get("type") == "image"]
        if msg_images:
            with st.expander("📊 Referenced Diagrams/Images"):
                cols = st.columns(min(len(msg_images), 3))
                for idx, s in enumerate(msg_images):
                    img_path = s.get("image_path")
                    if img_path and os.path.exists(img_path):
                        with cols[idx % 3]:
                            st.image(img_path, caption=f"Page {s.get('page', '?')} - {s.get('file', '')}", use_container_width=True)

col_query, col_viz = st.columns([4, 1])
with col_query:
    prompt = st.chat_input("Ask about your documents...")
with col_viz:
    visualize_mode = st.toggle("📊 Viz Mode", value=False, help="Generate chart from retrieved data")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base with The Harness..."):
            docs, dists = query_multimodal(prompt)
            
            if visualize_mode:
                with st.spinner("Analyzing data for visualization..."):
                    viz_code = generate_visualization_code(docs)
                    
                    if viz_code == "NO_DATA":
                        st.warning("The knowledge source contains no valid quantitative data for visualization")
                    else:
                        st.code(viz_code, language="python")
                        try:
                            exec(viz_code.replace("st.plotly_chart(", "st.empty()"))
                        except Exception as e:
                            st.error(f"Execution error: {str(e)}")
            
            if docs:
                full_prompt = build_multimodal_prompt(prompt, docs, dists)
            else:
                full_prompt = f"I don't have enough information in the document to answer this question.\n\nQuestion: {prompt}"

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
            
            has_images = any(s["type"] == "image" for s in sources)
            
            if has_images:
                st.markdown("### 📊 Diagrams from Document")
                img_sources = [s for s in sources if s["type"] == "image"]
                cols = st.columns(min(len(img_sources), 2))
                for idx, s in enumerate(img_sources):
                    img_path = s.get("image_path")
                    if img_path and os.path.exists(img_path):
                        with cols[idx % 2]:
                            st.image(img_path, caption=f"Page {s.get('page', '?')}", use_container_width=True)
            
            st.markdown("### 💡 Answer")
            st.markdown(answer)
            
            if sources:
                with st.expander("📚 Sources"):
                    text_sources = [s for s in sources if s["type"] != "image"]
                    for s in text_sources[:5]:
                        st.markdown(f"**Page {s.get('page', 0)}**: {s['text'][:150]}...")
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })