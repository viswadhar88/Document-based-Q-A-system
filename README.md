# 📚 AskYourDoc — Retrieval-Augmented Generation System for Multi-Document QA

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Python](https://img.shields.io/badge/Python-3.9+-green.svg)
![React](https://img.shields.io/badge/React-18.0+-61DAFB.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)

**Ask questions from your own uploaded documents — get accurate, document-grounded answers using RAG.**

---

## 🎯 Overview

**AskYourDoc** is a production-ready **Retrieval-Augmented Generation (RAG)** system that enables intelligent question-answering strictly over user-uploaded documents. Upload PDFs, Word docs, or text files and ask questions in plain natural language — the system retrieves semantically relevant content from your documents and generates accurate, context-aware answers using an LLM. No hallucination. No outside knowledge. Only your documents.

### ✨ Key Highlights

- **📄 Document-Grounded Answers** — LLM is strictly locked to uploaded document content only
- **🔍 Semantic Search** — BAAI/bge-small-en-v1.5 embeddings with FAISS vector search
- **⚡ Real-Time Progress** — WebSocket-based live upload and processing updates
- **🤖 Multi-LLM Support** — Groq (primary) with OpenAI automatic fallback
- **💬 Conversation Context** — Session-based chat with follow-up question support
- **📊 Analytics Dashboard** — Query trends, LLM usage, and system health monitoring
- **🧪 RAG Evaluation** — Automated 5-metric quality evaluation framework

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      Frontend (React)                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐    │
│  │  Upload  │  │   Chat   │  │Documents │  │ Analytics  │    │
│  └──────────┘  └──────────┘  └──────────┘  └────────────┘    │
└─────────────────────┬────────────────────────────────────────┘
                      │ HTTPS / WebSocket
┌─────────────────────▼────────────────────────────────────────┐
│                   Backend (FastAPI)                          │
│  ┌────────────────────────────────────────────────────────┐  │
│  │                     API Routes                         │  │
│  │   /documents   /query   /analytics   /ws   /system     │  │
│  └────────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────┐  │
│  │                   RAG Pipeline                         │  │
│  │ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐    │  │
│  │ │Document  │→│ Chunking │→│Embedding │→│  FAISS   │    │  │
│  │ │Processor │ │ Service  │ │ Service  │ │  Store   │    │  │
│  │ └──────────┘ └──────────┘ └──────────┘ └──────────┘    │  │
│  │                        ↓                               │  │
│  │                ┌──────────────┐                        │  │
│  │                │ LLM Service  │                        │  │
│  │                │(Groq→OpenAI) │                        │  │
│  │                └──────────────┘                        │  │
│  └────────────────────────────────────────────────────────┘  │
└─────────────────────┬────────────────────────────────────────┘
                      │
         ┌────────────▼────────────┐
         │       PostgreSQL        │
         │  ┌───────────────────┐  │
         │  │  Documents Table  │  │
         │  │  Analytics Table  │  │
         │  │   Query History   │  │
         │  │   Chat Sessions   │  │
         │  └───────────────────┘  │
         └─────────────────────────┘
```

---

## 🚀 Features

### Document Management
- **Multi-format Support** — PDF, DOCX, TXT, HTML, Markdown
- **Batch Upload** — Multiple documents processed simultaneously
- **Real-Time Progress** — WebSocket stage-by-stage progress (extract → chunk → embed → index)
- **50MB File Limit** — Enforced with graceful error handling
- **Preview & Download** — View and download uploaded documents

### Intelligent Search &Q&A
- **Semantic Search** — BGE embeddings find meaning, not just keywords
- **Document-Grounded Answers** — LLM strictly answers from retrieved content only
- **Typo Tolerance** — Understands misspelled queries without mentioning errors
- **Conversation Context** — Last 3 Q&A pairs passed for follow-up question handling
- **Source Attribution** — Every answer includes document source and similarity score

### RAG Pipeline
- **Smart Chunking** — 1000-character chunks with 200-character overlap
- **BGE Embeddings** — BAAI/bge-small-en-v1.5 (384-dim, normalized)
- **FAISS Vector Store** — IndexFlatIP with cosine similarity search
- **Filtered Search** — Per-document search using subset FAISS index
- **Multi-LLM Fallback** — Groq primary → OpenAI fallback, automatic routing

### Analytics & Evaluation
- **Usage Statistics** — Total queries, response times, success rates
- **Popular Questions** — Track frequently asked questions
- **Query Trends** — Visualize usage patterns over time
- **RAG Evaluation** — 5-metric automated quality framework

---

## 📊 RAG Evaluation Results

Evaluated using BAAI/bge-small-en-v1.5 embeddings:

| Metric | Score |
|--------|-------|
| Index Coverage |  100.0% | 
| Embedding Hit Rate |  100.0% | 
| Chunk Retrieval Quality |  0.611 | 
| Context Utilization |  100.0% | 
| Query Success Rate |  100.0% | 

---

## 🛠️ Tech Stack

### Backend
- **Framework** — FastAPI 0.104+
- **Database** — PostgreSQL with SQLAlchemy ORM
- **Vector Store** — FAISS (CPU/GPU)
- **Embedding Model** — BAAI/bge-small-en-v1.5 via SentenceTransformers
- **Document Processing** — PyPDF2, python-docx, BeautifulSoup4
- **Real-time** — WebSockets
- **LLM APIs** — Groq (llama-3.1-8b-instant), OpenAI (gpt-3.5-turbo-0125)

### Frontend
- **Framework** — React 18 with Vite
- **State Management** — Zustand
- **Styling** — Tailwind CSS
- **HTTP Client** — Axios with interceptors
- **Real-time** — WebSocket client with auto-reconnect
- **Notifications** — react-hot-toast
- **Animations** — Framer Motion
- **UI Components** — Custom glass-morphism design

### AI Models & Services
- **Embedding** — BAAI/bge-small-en-v1.5 (384 dimensions, normalized)
- **Primary LLM** — Groq llama-3.1-8b-instant
- **Fallback LLM** — OpenAI gpt-3.5-turbo-0125

---

## 📋 Prerequisites

- Python 3.9+
- Node.js 18+
- PostgreSQL 13+
- 4GB+ RAM
- CUDA 11.8+ (optional, for GPU acceleration)

---

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/AskYourDoc.git
cd AskYourDoc
```

### 2. Backend Setup

```bash
cd Backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env with your configuration

# Initialize database
python -c "from app.database import init_db; init_db()"

# Run the backend
uvicorn app.main:app --reload
```

Backend will be available at `http://localhost:8000`

### 3. Frontend Setup

```bash
cd Frontend

# Install dependencies
npm install

# Setup environment variables
cp .env.example .env
# Edit .env with your API URL

# Run development server
npm run dev
```

Frontend will be available at `http://localhost:5173`

---

## ⚙️ Environment Variables

### Backend (.env)

```env
# Database
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
POSTGRES_SERVER=localhost
POSTGRES_PORT=5432
POSTGRES_DB=document_qa

# LLM Providers (at least one required)
GROQ_API_KEY=gsk_...
OPENAI_API_KEY=sk-...

# Embedding Model
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

# GPU
USE_GPU=false

# CORS
CORS_ORIGINS=http://localhost:5173,http://localhost:3000

# File Upload
MAX_FILE_SIZE=20971520
ALLOWED_EXTENSIONS=.pdf,.docx,.txt,.md,.html
```

### Frontend (.env)

```env
VITE_API_URL=http://localhost:8000
```

---

## 📖 API Documentation

### Document Management

```
POST   /api/v1/documents/upload-with-progress   # Upload with WebSocket progress
GET    /api/v1/documents/                        # List all documents
GET    /api/v1/documents/{id}                    # Get specific document
DELETE /api/v1/documents/{id}                    # Delete document
POST   /api/v1/documents/search                  # Semantic search
GET    /api/v1/documents/{id}/content            # Get document text content
GET    /api/v1/documents/{id}/preview            # Preview document
GET    /api/v1/documents/{id}/download           # Download document
```

### Query & Chat

```
POST   /api/v1/query/ask                         # Ask a question
POST   /api/v1/query/ask-with-context            # Ask with conversation history
GET    /api/v1/query/history                     # Get query history
GET    /api/v1/query/sessions                    # Get all chat sessions
GET    /api/v1/query/sessions/{id}               # Get specific session
DELETE /api/v1/query/sessions/{id}               # Delete session
```

### Analytics

```
GET    /api/v1/analytics/stats                   # Usage statistics
GET    /api/v1/analytics/popular-questions        # Popular queries
GET    /api/v1/analytics/query-trends            # Query trends
GET    /api/v1/analytics/llm-usage               # LLM provider usage
```

### WebSocket

```
WS     /ws/client/{client_id}                    # Real-time updates
```

---

### Example: Ask a Question

```python
import requests

# Upload a document
with open("document.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/v1/documents/upload",
        files={"file": f}
    )
doc_id = response.json()["id"]

# Ask a question
response = requests.post(
    "http://localhost:8000/api/v1/query/ask",
    json={
        "question": "What is Newton's First Law?",
        "document_ids": [doc_id],
        "top_k": 5,
        "score_threshold": 0.3
    }
)

print(response.json()["answer"])
```

---

## 🎮 Usage Guide

### 1. Upload Documents
- Go to the Upload section
- Drag & drop or select files (PDF, DOCX, TXT, HTML, Markdown)
- Watch real-time progress as documents are processed
- Max size: 50MB per file

### 2. Ask Questions
- Go to the Chat section
- Type your question in natural language — typos are fine
- Optionally select specific documents to search within
- Get document-grounded answers with source references

### 3. View Analytics
- Check the Analytics section for system insights
- Monitor query volume, response times, and LLM usage
- Track popular questions across sessions

---

## ⚡ Performance Benchmarks

| Metric | Local (Intel Iris Xe / CPU) | Cloud (Shared CPU) | Speedup |
|--------|----------------------------|-------------------|---------|
| Embedding Generation | 444 ms | 2.1 s | ~4.7x Faster |
| Vector Search (FAISS) | 20 ms | 145 ms | ~7.3x Faster |
| Total Query Time | 1.4 s | 3.8 s | ~2.7x Faster |
| Document Throughput | 45 docs/min | 8–10 docs/min | ~4.5x Higher |

---

## 🧪 Running RAG Evaluation

```bash
cd Backend
PYTHONPATH=/path/to/Backend python3 app/rag_eval.py
```

Results are printed to terminal and saved to `app/rag_eval_results.json`

---

## 🐛 Troubleshooting

### Common Issues

1. **Embedding model slow to load**
   - First run downloads the model — subsequent runs use cached version
   - Ensure `EMBEDDING_MODEL=BAAI/bge-small-en-v1.5` is set in your `.env`

2. **PostgreSQL connection failed**
   - Verify PostgreSQL is running
   - Check credentials in `.env` match your PostgreSQL setup

3. **Groq API not responding**
   - System automatically falls back to OpenAI
   - Verify `GROQ_API_KEY` is valid in `.env`

4. **FAISS index empty after restart**
   - Call `POST /api/v1/documents/reset-vector-store` to rebuild the index
   - All previously uploaded documents will be re-embedded automatically

---

## 🔭 Future Scope

- Multi-modal support — images, tables, and scanned PDFs via OCR
- Cross-encoder re-ranking layer on top of FAISS for higher precision
- Token-by-token streaming from LLM to frontend via WebSocket
- User authentication with role-based document access control
- Docker containerization and cloud deployment pipeline
- Domain-specific embedding model fine-tuning

---

## 🙏 Acknowledgments

- BAAI for the BGE embedding models
- Meta AI for FAISS
- Groq for fast LLM inference
- FastAPI for the web framework
- Sentence Transformers for the embedding library

---

## 📞 Contact

**K Venkata Viswadhar**

GitHub: [@viswadhar88]([https://github.com/viswadhar88](https://github.com/viswadhar88))

Project Link: [https://github.com/viswadhar88/Document-based-Q-A-system](https://github.com/viswadhar88/Document-based-Q-A-system)

---

