# Local RAG: Multimodal Retrieval-Augmented Generation

A powerful local-first RAG system that processes PDF documents with text, tables, and images using Ollama for embeddings and inference.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Local RAG Pipeline                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌──────────────────┐    ┌───────────────────────┐ │
│  │   PDF Files  │───▶│ Contextual       │───▶│ Chroma Vector Store  │ │
│  │   (Data/)    │    │ Retrieval        │    │ (Embeddings)         │ │
│  └──────────────┘    └──────────────────┘    └───────────────────────┘ │
│                              │                          │              │
│                              ▼                          ▼              │
│                       ┌─────────────┐          ┌───────────────────┐  │
│                       │ Docling     │          │ Similarity        │  │
│                       │ (OCR/Layout)│          │ Search (k=20)     │  │
│                       └─────────────┘          └───────────────────┘  │
│                                                         │              │
│                                                         ▼              │
│  ┌──────────────────┐    ┌─────────────────┐    ┌───────────────────┐ │
│  │  The Harness    │◀───│ Multi-Query     │◀───│ Retrieved         │ │
│  │  (LLM Response)  │    │ Strategy        │    │ Context           │ │
│  └──────────────────┘    └─────────────────┘    └───────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Features

### 1. **Multimodal Document Processing**
- **Docling** for advanced PDF parsing with OCR and layout understanding
- Extracts text, tables, and images from PDFs
- Falls back to PyPDF for basic text extraction if Docling fails

### 2. **Contextual Retrieval**
- Generates context for each chunk to improve retrieval accuracy
- Uses LLM to describe how a chunk fits into the broader document
- Enriches chunks with contextual information before embedding

### 3. **Three-Layer Intelligence Engine**

| Layer | Component | Description |
|-------|-----------|-------------|
| **Ingestion** | Contextual Retrieval | Enriches document chunks with context |
| **Core** | The Harness | Friendly teaching assistant with Plan-Route-Act-Verify |
| **Viz** | Data Analyst | Generates Plotly visualizations from data |

### 4. **Smart Retrieval**
- Detects visual queries (diagrams, charts, graphs)
- Returns relevant images, text, and tables based on query type
- Uses cosine similarity with configurable threshold (0.9)

### 5. **Vision-Language Model**
- Uses **LLaVA** for image captioning and analysis
- Extracts visual elements from diagrams
- Provides detailed descriptions for images in documents

### 6. **Context Collapse**
- Automatically summarizes conversation when approaching token limits
- Keeps essential information for continuing the dialogue

## Requirements

- **Ollama** (running locally on port 11434)
- Models:
  - `nomic-embed-text` (embeddings)
  - `phi3` (LLM)
  - `llava` (vision model)
- Python dependencies (see `requirements.txt`)

## Installation

```bash
# Clone the repository
git clone https://github.com/skshm-cyber/local-rag.git
cd local-rag

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Ollama and pull required models
ollama pull nomic-embed-text
ollama pull phi3
ollama pull llava
```

## Usage

### 1. Start Ollama (if not running)
```bash
ollama serve
```

### 2. Run the Streamlit App
```bash
streamlit run app/app.py
```

### 3. Using the Interface

1. **Add PDFs**: Place your PDF files in the `data/` folder
2. **Sync**: Click "🔄 Sync Folder" to process and index documents
3. **Query**: Ask questions about your documents in the chat
4. **Visualize**: Toggle "📊 Viz Mode" to generate charts from data

## Configuration

Key parameters in `app/multimodal_rag.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EMBEDDING_MODEL` | nomic-embed-text | Embedding model |
| `LLM_MODEL` | phi3 | Language model |
| `VISION_MODEL` | llava | Vision model |
| `RETRIEVAL_K` | 20 | Number of chunks to retrieve |
| `CHUNK_SIZE` | 400 | Text chunk size |
| `MAX_COSINE_DIST` | 0.9 | Similarity threshold |

## Project Structure

```
local-rag/
├── app/
│   ├── __init__.py
│   ├── app.py              # Streamlit UI
│   ├── multimodal_rag.py   # Core RAG logic
│   └── data/               # Sample PDF
├── data/                   # PDF documents
├── chroma_db/              # Vector database
├── extracted_images/       # Extracted images
├── requirements.txt        # Python dependencies
└── README.md
```

## How It Works

### Document Processing Pipeline
1. PDFs are loaded from `data/` folder
2. Docling extracts text, tables, and images
3. Each text chunk gets contextual enrichment
4. Images are captioned using LLaVA
5. All content is embedded and stored in Chroma

### Query Flow
1. User enters a question
2. System detects if it's a visual query
3. Retrieves relevant text, images, and tables
4. The Harness generates a friendly response
5. Referenced images are displayed

## Troubleshooting

- **Ollama not available**: Ensure Ollama is installed and running
- **No models found**: Pull required models with `ollama pull <model-name>`
- **Empty responses**: Try syncing documents first
- **Slow performance**: Reduce `RETRIEVAL_K` or `CHUNK_SIZE`

## License

MIT