# Approach & Design Decisions

## Chunking Strategy

- **Method:**  
  Use PDF **page-level extraction** via **PyMuPDF** (preserves page boundaries).  
  Detect section headers heuristically — e.g., **Abstract**, **Introduction**, **Methods**, **Results**, **Conclusion**, **References**.

- **Chunking logic:**  
  - Chunk by **paragraphs** with a **maximum window** of **1800 characters**.  
  - Use **200-character overlap** to preserve local context and continuity.  
  - Further split large chunks at **sentence boundaries** to avoid mid-sentence truncation.

- **Rationale:**  
  Preserves semantic context, reduces retrieval noise, and ensures chunks are small enough to fit within LLM prompt budgets.

---

## Embedding Model

- **Model:** `sentence-transformers/all-mpnet-base-v2`  
  - Strong semantic retrieval performance on academic text.  
  - Produces **768-dimensional** compact vectors.

- **Trade-offs:**  
  Larger models (e.g., **Longformer-based**) can capture longer contexts but are **heavier to serve** and slower to embed.

---

## Prompt Engineering

- **Pipeline behavior:**  
  - Sends **top-k retrieved chunks** to the LLM as the **only context**.  
  - The prompt instructs the LLM to:
    - Use **only provided excerpts**.  
    - Return a **JSON structure** containing:
      - `answer`
      - `citations`
      - `confidence`

- **Benefits:**  
  Minimizes hallucinations and ensures **reliable citation extraction**.

---

## Database Schema

| Table | Purpose |
|--------|----------|
| **papers** | Stores basic metadata and filename (for deduplication). |
| **chunks** | Stores text, page range, section, and Qdrant vector ID. |
| **query_history** | Tracks user queries, referenced papers, response times, and summary results. |

- **Rationale:**  
  Store **minimal metadata** for flexibility and allow efficient **deletion and tracking**.

- **Trade-off:**  
  Chunk text is stored both in the **DB** and as **payload in Qdrant** for easy retrieval.  
  If storage is limited, only references should be stored in Qdrant.

---

## Trade-offs & Limitations

- **Section detection:**  
  Heuristic-based; may fail for **non-standard formats**. Could be improved using **layout parsers** or **ML-based segmentation**.

- **Ollama compatibility:**  
  Request/response structure may need updates depending on your installed Ollama version.

- **Performance considerations:**  
  - Goal: process **5 papers in < 2 minutes**.  
  - Depends on machine specs, embedding model (CPU/GPU), and PDF size.  
  - **Use GPU** for embedding to improve speed.

---
