import time
import json
import logging
from typing import List, Dict, Any, Optional

from src.services.embedding_service import get_embedding_service
from src.services.qdrant_client import qdrant_service
from src.services.llm_client import get_llm_client, OllamaError  # if you raised OllamaError earlier
from src.models.db import SessionLocal
from src.models.orm_models import QueryHistory
from src.config import settings

logger = logging.getLogger(__name__)


def build_prompt(question: str, contexts: List[Dict[str, Any]]) -> str:
    ctx_texts = []
    for c in contexts:
        header = f"[{c.get('paper_title','unknown')}] Section: {c.get('section')} (page {c.get('page')})"
        ctx_texts.append(header + "\n" + c.get("text", ""))
    context_block = "\n\n---\n\n".join(ctx_texts) if ctx_texts else "No context available."
    prompt = f"""
You are a helpful assistant. Answer the question using the provided paper excerpts and return ONLY a single JSON object (no commentary) with these keys:
- answer: string (Provide a concise answer)
- citations: list of objects [{"{"}paper_title: str, section: str, page: int, relevance_score: float{"}"}]
- sources_used: list[str] (filenames)
- confidence: float (0.0-1.0)

Context:
{context_block}

Question: {question}
Use only the provided excerpts — do NOT invent sources. If the answer is not contained in the excerpts, return an empty answer object as specified below.
"""
    return prompt.strip()


def run_query(question: str, top_k: int = 5, paper_ids: Optional[List[int]] = None) -> Dict[str, Any]:
    start = time.time()
    sources = set()
    contexts: List[Dict[str, Any]] = []
    answer = ""
    citations = []
    confidence = None

    if not question or not isinstance(question, str):
        raise ValueError("question must be a non-empty string")

    # 1) Embed the question (use the service wrapper which has its own error handling)
    try:
        emb = get_embedding_service()
        vecs = emb.embed_texts([question], normalize=True)  # returns list[list[float]]
        if not vecs:
            raise RuntimeError("Embedding service returned no vectors")
        vec = vecs[0]
    except Exception as e:
        logger.exception("Embedding failed: %s", e)
        # fail early, record duration and return error-like response
        duration = (time.time() - start) * 1000.0
        return {
            "error": "embedding_failed",
            "message": str(e),
            "response_time_ms": duration
        }

    # 2) Search qdrant
    try:
        hits = qdrant_service.search(vec, top_k=top_k)
    except Exception as e:
        logger.exception("Qdrant search failed: %s", e)
        duration = (time.time() - start) * 1000.0
        return {"error": "qdrant_search_failed", "message": str(e), "response_time_ms": duration}

    # 3) Build contexts from hits (safe extraction & truncation)
    for h in hits or []:
        try:
            payload = getattr(h, "payload", {}) or {}
            contexts.append({
                "paper_title": payload.get("paper_title") or payload.get("title") or "unknown",
                "section": payload.get("section") or "Body",
                "page": payload.get("page_start") or payload.get("page") or None,
                "text": (payload.get("text") or "")[:2000],
                "relevance": float(getattr(h, "score", 0.0))
            })
            sources.add(payload.get("paper_filename") or payload.get("filename"))
        except Exception:
            # skip malformed hit but log for debugging
            logger.exception("Malformed hit encountered: %s", h)

    # 4) Build prompt and call LLM
    prompt = build_prompt(question, contexts)
    try:
        llm = get_llm_client()
        llm_resp = llm.generate(prompt)
    except OllamaError as e:
        logger.exception("LLM call failed (OllamaError): %s", e)
        duration = (time.time() - start) * 1000.0
        return {"error": "llm_error", "message": str(e), "response_time_ms": duration}
    except Exception as e:
        logger.exception("LLM call failed: %s", e)
        duration = (time.time() - start) * 1000.0
        return {"error": "llm_error", "message": str(e), "response_time_ms": duration}

    # 5) Parse LLM response: robust parsing depending on client format
    parsed = None
    try:
        # If the client returns a dict with 'raw' (per updated OllamaClient), prefer that.
        if isinstance(llm_resp, dict):
            if "raw" in llm_resp:
                # raw may be a dict or list; try to pull string output
                raw = llm_resp["raw"]
                # attempt to find a text field
                if isinstance(raw, dict):
                    candidate = raw.get("text") or raw.get("output") or json.dumps(raw)
                else:
                    candidate = json.dumps(raw)
            elif "raw_text" in llm_resp:
                candidate = llm_resp["raw_text"]
            else:
                candidate = str(llm_resp)
        else:
            candidate = str(llm_resp)

        # try parse candidate as JSON
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            answer = parsed.get("answer") or parsed.get("text") or candidate
            citations = parsed.get("citations") or []
            confidence = parsed.get("confidence")
        else:
            answer = candidate
            citations = []
            confidence = None
    except Exception:
        # fallback: use whatever llm_resp contains
        logger.debug("Failed to parse LLM response as JSON; using raw response.")
        answer = str(llm_resp)
        citations = []
        confidence = None

    # 6) Store query history safely
    duration = (time.time() - start) * 1000.0
    db = SessionLocal()
    try:
        qh = QueryHistory(
            query_text=question,
            papers_referenced=[s for s in sources if s],
            response_time_ms=duration,
            result_summary=(answer[:2000] if answer else None)
        )
        db.add(qh)
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("Failed to write QueryHistory to DB")
    finally:
        db.close()

    return {
        "answer": answer,
        "citations": citations,
        "sources_used": [s for s in sources if s],
        "confidence": float(confidence) if confidence is not None else 0.5,
        "response_time_ms": duration
    }
