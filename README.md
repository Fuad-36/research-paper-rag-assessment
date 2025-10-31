# 🎓 Research Paper RAG System

A production-ready Retrieval-Augmented Generation (RAG) system for querying academic research papers using Python, Fast API, PostgreSQL, Qdrant, and Ollama.

## 🚀 Features

- ✅ **Automated PDF Processing** - Upload PDFs, extract text, metadata, and sections using Gemini LLM
- ✅ **Intelligent Chunking** - Semantic chunking with section awareness and overlap
- ✅ **Vector Search** - 384-dimensional embeddings with Qdrant vector database
- ✅ **RAG Query System** - Context-aware answers with citations and sources
- ✅ **Analytics Dashboard** - Query history, popular topics, and system metrics
- ✅ **RESTful API** - Clean MVC architecture with comprehensive endpoints

## 📋 Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| **Vector DB** | Qdrant | Fast similarity search |
| **Database** | PostgreSQL | Metadata & query history |
| **LLM** | Ollama | Answer generation |
| **Embeddings** | sentence-transformers | Text vectorization |
| **Backend** | Python + FastAPI | API service |


## 🏗️ Architecture

```
+-------------------+
|   Client (API)    |
| curl / Postman /  |
| Python requests    |
+---------+----------+
          |
          v
+---------------------------+
|       FastAPI Backend     |
| (API + Orchestration)     |
+-----------+---------------+
            |
     ┌──────┴─────────────────────────────────────────────────────────────┐
     │                                                                    │
     v                                                                    v
+---------------------+         +-------------------------+      +----------------+
| PDF Processor       |         | Embedding Service       |      | Qdrant Vector  |
| (section detection, |         | (Sentence Transformers) |      | Store          |
| chunking)           |         +-------------------------+      | (semantic search)|
+---------------------+                                         +----------------+
          |                                                            |
          v                                                            |
+---------------------------+                                           |
| PostgreSQL Database       |<------------------------------------------+
| (papers, chunks, history) |
+---------------------------+
          |
          v
+--------------------+
|  Ollama LLM        |
| (Answer synthesis) |
+--------------------+

```

## 📁 Project Structure

```
research-paper-rag-assessment/
├── sample_papers/                  # Example research papers for testing ingestion
│
├── src/                            # Main source code
│   ├── api/                        # API layer (FastAPI routes & endpoints)
│   │   ├── __init__.py
│   │   └── routes.py               # Defines REST endpoints
│   │   
│   │
│   ├── models/                     # Database models & ORM schema
│   │   ├── db.py                   # Database connection setup
│   │   └── orm_models.py           # SQLAlchemy ORM models
│   │   
│   │
│   ├── services/                   # Core business logic layer
│   │   ├── embedding_service.py    # Embedding generation (SBERT - all-mpnet-base-v2)
│   │   ├── llm_client.py           # LLM interface (Ollama / Gemini)
│   │   ├── pdf_processor.py        # PDF parsing, text extraction, and chunking
│   │   ├── qdrant_client.py        # Qdrant vector DB client (embedding storage & search)
│   │   └── rag_pipeline.py         # RAG orchestration: retrieval + generation
│   │   
│   │
│   ├── config.py                   # App configuration and environment variables
│   └── main.py                     # FastAPI app entry point
│
├── uploaded_papers/                # User-uploaded PDFs 
│
├── test_queries.json               # Example query inputs for testing
├── requirements.txt                # Python dependencies
├── .env.example                    # Sample environment variable file
│
├── README.md                       # Project documentation
├── APPROACH.md                     # System design & methodology
├── SUBMISSION_GUIDE.md             # Instructions for submission/evaluation
├── PULL_REQUEST_TEMPLATE.md        # PR template (for collaboration)
├── .gitignore                      # Git ignore rules
└── venv/                           # Python virtual environment (local)

```

## 🚀 Quick Start

### Prerequisites

- Python 3.10 <= version < 3.13
- PostGRE SQL
- Docker (for Qdrant)
- Ollama with llama3:latest

### Installation

```bash
#  Clone repository
git clone <your-repo-url>
cd research-paper-rag-assessment

# Create virtual environment and activate
python -3.11 -m venv venv  
.\venv\Scripts\Activate

# Upgrade core tools
python -m pip install --upgrade pip setuptools wheel

# Install dependencies
python -m pip install -r requirements.txt

# Sometimes, the package fitz can conflict with PyMuPDF.
pip uninstall fitz PyMuPDF -y
pip install PyMuPDF

# Run qdrant with docker
docker run -p 6333:6333 qdrant/qdrant

# Start the ollama
ollama serve

# Run the application
uvicorn src.main:app --reload
```



## 📡 API Endpoints

### Papers API

#### Upload Paper
```bash
POST /api/papers/upload
Content-Type: multipart/form-data

file: <PDF file>
```

#### List Papers
```bash
GET /api/papers
```

#### Get Paper Details
```bash
GET /api/papers/:id
```

#### Delete Paper
```bash
DELETE /api/papers/:id
```

#### Get Paper Stats
```bash
GET /api/papers/:id/stats
```

### Query API

#### Query Papers
```bash
POST /api/query
Content-Type: application/json

{
  "question": "What is machine learning?",
  "top_k": 5,
  "paper_ids": ["optional"]
}
```

####  Paper Management
```python
GET    /api/papers              # List all papers
GET    /api/papers/{id}         # Get paper details
DELETE /api/papers/{id}         # Remove paper + vectors
GET    /api/papers/{id}/stats   # View/download stats
```

####  Query History & Analytics
```python
GET /api/queries/history         # Recent queries
GET /api/analytics/popular       # Most queried topics
```




## 📊 Example Usage

### Upload a Paper

```bash
curl -X POST http://localhost:8000/api/papers/upload \
  -F "file=@paper.pdf"
```

### Upload Multiple Papers

```bash
curl -X POST http://localhost:8000/api/papers/upload \
  -F "file=@paper_1.pdf" -F "file=@paper_2.pdf"
```

### Ask a Question

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What methodology was used?",
    "top_k": 5
  }'
```

### Get Analytics

```bash
curl http://localhost:3000/api/analytics
```

### ⚙️ Component Breakdown

1. **FastAPI Backend:** 
- Core API layer exposing endpoints for:
- Paper upload and processing (POST /api/papers/upload)
- Query answering (POST /api/query)
- Paper management and analytics
- Coordinates the entire RAG pipeline.

2. **PDF Processor**

- Uses PyMuPDF to extract structured text with section awareness (Abstract, Methodology, etc.).
- Splits text into semantic chunks (≈1800 characters with 200 overlap).
- Extracts metadata such as title, authors, year, and page numbers.

3. **Embedding Service**
- Powered by sentence-transformers to create dense vector embeddings for both paper chunks and user queries.
- Ensures consistent embedding space for retrieval.

4. **Qdrant Vector Database**

- Stores embeddings with metadata (paper ID, section, page, relevance).
- Performs vector similarity search to retrieve the most relevant chunks per query.
- Enables filtering by paper IDs or sections.

5. **PostgreSQL Database**

- Stores structured metadata:
    - Paper details (title, authors, year)
    - Chunk info (section, page range)
    - Query history (text, response time, referenced papers)

- Maintains consistency and supports analytics endpoints.

6. **Ollama (LLM)**

- Synthesizes final answers using retrieved chunks as context.
- Prompted to return structured JSON containing:
  - answer
  - citations (with title, section, page)
  - confidence score

## 🔒 Error Handling

- Validation on all inputs
- Graceful fallbacks
- Detailed error logging
- User-friendly error messages





## 👥 Authors

- MD Fuad Al Amin - Initial work

## 🙏 Acknowledgments

- UpscaleBD for giving the assignment
- Open AI for assistance
- Ollama for LLM capabilities
- Qdrant team for vector database

