import os
import io
import base64
import time
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import streamlit as st
from PIL import Image
from langchain_core.documents import Document

OLLAMA_HOST = "http://127.0.0.1:11434"
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data"
IMAGES_PATH = PROJECT_ROOT / "extracted_images"
CHROMA_PATH = PROJECT_ROOT / "chroma_db"

EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "phi3"
VISION_MODEL = "llava"

RETRIEVAL_K = 20
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_COSINE_DIST = 0.9
MAX_CONTEXT_CHARS = 8000

IMAGES_PATH.mkdir(exist_ok=True)

OLLAMA_AVAILABLE = False

# Contextual retrieval disabled for fast offline processing

HARNESS_SYSTEM = """You are a friendly and clear teaching assistant. Your job is to explain concepts simply and show relevant images when available.

Guidelines:
- Explain concepts in simple, easy-to-understand language
- Use short sentences and clear examples
- When images are available, describe what they show
- If you don't have enough information, say so simply
- Always cite the source page number for reference

Answer the user's question based ONLY on the provided context."""

VISUALIZATION_SYSTEM = """Role: You are a Data Analyst Agent. Your goal is to generate Python code for plotly.express to visualize the provided structured data.

Instructions:
- Use ONLY the data values extracted from the retrieved document chunks.
- Do not interpolate, estimate, or "hallucinate" missing data points.
- Generate a complete script using the st.plotly_chart() function for Streamlit.
- If the source data does not contain quantitative values suitable for a chart, respond with: "The knowledge source contains no valid quantitative data for visualization"."""

def check_ollama() -> bool:
    global OLLAMA_AVAILABLE
    try:
        import requests
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=2)
        OLLAMA_AVAILABLE = resp.status_code == 200
    except:
        OLLAMA_AVAILABLE = False
    return OLLAMA_AVAILABLE


def get_text_embeddings():
    from langchain_ollama import OllamaEmbeddings
    return OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_HOST)


def get_llm():
    from langchain_ollama import ChatOllama
    return ChatOllama(model=LLM_MODEL, base_url=OLLAMA_HOST, temperature=0.1)


def get_vision_llm():
    from langchain_ollama import ChatOllama
    return ChatOllama(model=VISION_MODEL, base_url=OLLAMA_HOST, temperature=0.1)


@st.cache_resource
def get_vector_store():
    from langchain_chroma import Chroma
    return Chroma(
        persist_directory=str(CHROMA_PATH),
        embedding_function=get_text_embeddings(),
        collection_metadata={"hnsw:space": "cosine"},
    )


def parse_pdf_document(pdf_path: str) -> Tuple[List[Document], List[Tuple[Image.Image, str, int]]]:
    """Parse PDF completely offline using PyPDF."""
    from pypdf import PdfReader
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    images = []
    text_docs = []
    
    try:
        reader = PdfReader(pdf_path)
        for page_num, page in enumerate(reader.pages):
            for img_index, img in enumerate(page.images):
                try:
                    image_data = img.data
                    image = Image.open(io.BytesIO(image_data))
                    image_id = f"{Path(pdf_path).stem}_p{page_num+1}_img{img_index+1}"
                    images.append((image, image_id, page_num + 1))
                except Exception as e:
                    print(f"Error loading image on page {page_num+1}: {e}")
    except Exception as e:
        print(f"Image extraction error: {e}")
    
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        splits = splitter.split_documents(docs)
        
        file_name = Path(pdf_path).name
        for s in splits:
            s.metadata["source_file"] = file_name
            s.metadata["content_type"] = "text"
        text_docs = splits
    except Exception as e:
        print(f"Text extraction error: {e}")
    
    return text_docs, images


def process_pdf_document(pdf_path: str) -> Tuple[List[Document], List[Document]]:
    """Process PDF offline - extracts and chunks text, saves images."""
    check_ollama()
    
    text_docs, extracted_images = parse_pdf_document(pdf_path)
    
    image_docs = []
    for image, image_id, page_num in extracted_images:
        try:
            image_path = save_image(image, image_id)
            
            context = ""
            for doc in text_docs:
                if doc.metadata.get("page") == page_num:
                    context += doc.page_content[:500] + " "
            
            caption = caption_image_with_vlm(image, image_id, context.strip())
            
            doc = Document(
                page_content=caption,
                metadata={
                    "source_file": Path(pdf_path).name,
                    "content_type": "image",
                    "image_id": image_id,
                    "image_path": image_path,
                    "page": page_num,
                }
            )
            image_docs.append(doc)
        except Exception as e:
            print(f"Error processing image {image_id}: {e}")
    
    return text_docs, image_docs


def caption_image_with_vlm(image: Image.Image, image_id: str, context: str = "") -> str:
    """Generate caption using VLM if available."""
    if not check_ollama():
        return generate_caption_fallback(image, image_id)
    
    try:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode()
        
        prompt = f"""Describe this diagram/image from a PDF document.
Related text: {context}

Provide: 1) Type of visual 2) All elements 3) Key info conveyed."""

        vision_llm = get_vision_llm()
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        response = vision_llm.invoke(messages)
        return response.content if hasattr(response, "content") else str(response)
        
    except Exception as e:
        print(f"VLM error: {e}")
        return generate_caption_fallback(image, image_id)


def generate_caption_fallback(image: Image.Image, image_id: str) -> str:
    """Fallback caption when VLM unavailable."""
    width, height = image.size
    mode = image.mode
    
    caption = f"Image: {image_id}\nSize: {width}x{height}, Mode: {mode}\n"
    
    if OLLAMA_AVAILABLE:
        caption += "VLM available for detailed analysis"
    else:
        caption += "Install Ollama + llava model for detailed image analysis"
    
    return caption


def save_image(image: Image.Image, image_id: str) -> str:
    """Save image to file."""
    image_path = IMAGES_PATH / f"{image_id}.png"
    if not image_path.exists():
        image.save(image_path, "PNG")
    return str(image_path)


def get_image_from_path(image_path: str) -> Optional[Image.Image]:
    """Load image from path."""
    try:
        if os.path.exists(image_path):
            return Image.open(image_path)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
    return None


def analyze_image_with_vlm(image_path: str, question: str) -> str:
    """Analyze image with VLM if available."""
    if not check_ollama():
        img = get_image_from_path(image_path)
        if img:
            return f"VLM unavailable. Image info: {img.size}, {img.mode}. Install Ollama to analyze."
        return "Image not found"
    
    try:
        image = get_image_from_path(image_path)
        if not image:
            return "Image not found"
        
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode()
        
        prompt = f'Analyze this image. Question: "{question}"'
        
        vision_llm = get_vision_llm()
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        response = vision_llm.invoke(messages)
        return response.content if hasattr(response, "content") else str(response)
        
    except Exception as e:
        return f"Error: {str(e)}"


def sync_multimodal_data() -> Dict[str, int]:
    """Scan and index PDFs using Docling with contextual retrieval."""
    check_ollama()
    
    pdf_files = [f for f in os.listdir(DATA_PATH) if f.lower().endswith(".pdf")]
    
    stats = {"text_chunks": 0, "images": 0}
    
    if not pdf_files:
        return stats
    
    vs = get_vector_store()
    existing_sources = set()
    
    try:
        existing = vs.get()
        if existing and "metadatas" in existing:
            for meta in existing["metadatas"]:
                if "source_file" in meta:
                    existing_sources.add(meta["source_file"])
    except:
        pass
    
    for file_name in pdf_files:
        if file_name in existing_sources:
            continue
            
        file_path = str(DATA_PATH / file_name)
        
        with st.spinner(f"Processing {file_name}..."):
            text_docs, image_docs = process_pdf_document(file_path)
            
            if text_docs:
                vs.add_documents(text_docs)
                stats["text_chunks"] += len(text_docs)
            
            if image_docs:
                vs.add_documents(image_docs)
                stats["images"] += len(image_docs)
    
    return stats


def query_multimodal(query: str) -> Tuple[List[Document], List[float]]:
    """Query knowledge base with optimized parameters."""
    check_ollama()
    
    vs = get_vector_store()
    
    if vs._collection.count() == 0:
        return [], []
    
    query_lower = query.lower()
    visual_keywords = ['diagram', 'chart', 'graph', 'visual', 'show', 'figure', 'image', 'picture', 'flowchart', 'tree', 'example', 'step', 'how it work', 'digram', 'visualize', 'draw']
    is_visual_query = any(word in query_lower for word in visual_keywords) or 'how' in query_lower or 'what is' in query_lower
    
    if is_visual_query:
        results = vs.similarity_search_with_score(query, k=RETRIEVAL_K)
        all_docs = [(doc, dist) for doc, dist in results]
        
        image_docs = [d for d, _ in all_docs if d.metadata.get('content_type') == 'image']
        text_docs = [d for d, _ in all_docs if d.metadata.get('content_type') == 'text']
        table_docs = [d for d, _ in all_docs if d.metadata.get('content_type') == 'table']
        
        if image_docs:
            final_docs = image_docs[:15] + text_docs[:6] + table_docs[:3]
        else:
            search_terms = [
                "binary tree diagram structure",
                "tree data structure visualization", 
                "binary search tree example",
                "recursion tree flowchart",
                "algorithm tree diagram"
            ]
            all_images = []
            for term in search_terms:
                results_img = vs.similarity_search_with_score(term, k=8)
                img_by_page = [(d, s) for d, s in results_img if d.metadata.get('content_type') == 'image']
                all_images.extend(img_by_page)
            
            unique_imgs = []
            seen_pages = set()
            for d, s in all_images:
                page = d.metadata.get('page', 0)
                if page not in seen_pages:
                    seen_pages.add(page)
                    unique_imgs.append((d, s))
            
            final_docs = [d for d, _ in unique_imgs[:10]] + text_docs[:6]
    else:
        results = vs.similarity_search_with_score(query, k=RETRIEVAL_K)
        all_docs = [(doc, dist) for doc, dist in results if dist <= MAX_COSINE_DIST]
        
        text_docs = [d for d, _ in all_docs if d.metadata.get('content_type') == 'text']
        image_docs = [d for d, _ in all_docs if d.metadata.get('content_type') == 'image']
        table_docs = [d for d, _ in all_docs if d.metadata.get('content_type') == 'table']
        
        final_docs = image_docs[:6] + text_docs[:8] + table_docs[:2]
    
    final_docs = sorted(final_docs, key=lambda d: d.metadata.get('page', 0))
    
    return final_docs, [0.5] * len(final_docs)


def get_images_for_page(page_num: int, source_file: str = None) -> List[str]:
    """Get all image paths for a specific page."""
    import glob
    pattern = str(IMAGES_PATH / f"*{page_num}*.png")
    images = glob.glob(pattern)
    return images


def get_page_images(query: str) -> List[Tuple[str, int]]:
    """Get images related to query by page number."""
    vs = get_vector_store()
    results = vs.similarity_search_with_score(query, k=3)
    
    page_images = []
    for doc, dist in results:
        page = doc.metadata.get('page', 0)
        if page and page > 0:
            imgs = get_images_for_page(page)
            for img in imgs:
                page_images.append((img, page))
    
    return page_images[:6]


def harness_execute(query: str, docs: List[Document], dists: List[float]) -> str:
    """Execute the Harness - simple friendly answering."""
    check_ollama()
    llm = get_llm()
    
    text_docs = [d for d in docs if d.metadata.get("content_type") == "text"]
    image_docs = [d for d in docs if d.metadata.get("content_type") == "image"]
    table_docs = [d for d in docs if d.metadata.get("content_type") == "table"]
    
    context_parts = []
    for idx, doc in enumerate(docs):
        page = doc.metadata.get("page", "?")
        content_type = doc.metadata.get("content_type", "text")
        content = doc.page_content[:500]
        context_parts.append(f"[Page {page}] ({content_type}):\n{content}")
    
    context_str = "\n\n".join(context_parts)
    
    harness_prompt = f"""{HARNESS_SYSTEM}

QUESTION: {query}

CONTEXT FROM YOUR DOCUMENTS:
{context_str}

Provide a clear, simple answer. If images are available, mention them. If information is missing, say "I don't have that information from the documents"."""

    try:
        response = llm.invoke(harness_prompt)
        return response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        return f"Error: {str(e)}"


def generate_visualization_code(docs: List[Document]) -> str:
    """Generate Plotly visualization code from retrieved data."""
    check_ollama()
    llm = get_llm()
    
    data_docs = [d for d in docs if d.metadata.get("content_type") in ["text", "table"]]
    
    data_summary = ""
    for doc in data_docs:
        content = doc.page_content
        numbers = []
        import re
        num_matches = re.findall(r'\d+(?:\.\d+)?', content)
        if num_matches:
            numbers.extend([float(n) for n in num_matches[:20]])
        data_summary += f"Page {doc.metadata.get('page', '?')}: {content[:500]}\n"
    
    viz_prompt = f"""{VISUALIZATION_SYSTEM}

RETRIEVED DATA:
{data_summary}

Analyze the data above and determine if there are quantitative values suitable for visualization.
If yes, generate a complete Python script using plotly.express and st.plotly_chart().
If no, respond exactly with: "The knowledge source contains no valid quantitative data for visualization"

Generate code only (no explanations):"""
    
    try:
        response = llm.invoke(viz_prompt)
        result = response.content if hasattr(response, "content") else str(response)
        
        if "no valid quantitative data" in result.lower():
            return "NO_DATA"
        
        if "import" in result and "plotly" in result:
            return result
        
        return "NO_DATA"
    except Exception as e:
        return "NO_DATA"


def build_multimodal_prompt(query: str, docs: List[Document], dists: List[float]) -> str:
    """Build prompt that forces the LLM to use ONLY provided context."""
    text_docs = [d for d in docs if d.metadata.get("content_type") == "text"]
    image_docs = [d for d in docs if d.metadata.get("content_type") == "image"]
    table_docs = [d for d in docs if d.metadata.get("content_type") == "table"]
    
    prompt = """You are a factual question answering system about algorithms and data structures.
Answer ONLY using the context provided below. DO NOT make up any information.

"""

    if image_docs:
        prompt += "\n📊 DIAGRAMS/VISUALS FROM THE DOCUMENT:\n"
        for doc in image_docs:
            page = doc.metadata.get("page", "?")
            img_desc = doc.page_content
            prompt += f"[Page {page}]: {img_desc}\n"
    
    if table_docs:
        prompt += "\n📊 TABLES FROM DOCUMENT:\n"
        for doc in table_docs:
            page = doc.metadata.get("page", "?")
            prompt += f"[Page {page}]: {doc.page_content}\n"
    
    if text_docs:
        prompt += "\n📄 TEXT CONTENT FROM DOCUMENT:\n"
        for doc in text_docs[:8]:
            page = doc.metadata.get("page", "?")
            content = doc.page_content[:800]
            prompt += f"[Page {page}]: {content}\n"
    
    prompt += f"""

QUESTION: {query}

IMPORTANT: When describing algorithms, include the step-by-step process and reference the page numbers.
If there are diagrams available, describe what they show in detail.

ANSWER:"""

    return prompt


def context_collapse_summary(messages: List[Dict], max_chars: int = 4000) -> str:
    """Trigger context collapse summary when approaching token limit."""
    check_ollama()
    llm = get_llm()
    
    history = "\n".join([f"{m['role']}: {m['content'][:500]}" for m in messages[-10:]])
    
    collapse_prompt = f"""Summarize this conversation into a concise context保留关键信息用于后续对话。保留关键信息用于后续对话。Keep only essential info for continuing the conversation:

{history}

Provide a concise summary (under 500 words):"""
    
    try:
        response = llm.invoke(collapse_prompt)
        return response.content if hasattr(response, "content") else str(response)
    except:
        return history