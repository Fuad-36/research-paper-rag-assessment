from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from src.services.pdf_processor import extract_pdf, chunk_text_with_sections
from src.services.embedding_service import get_embedding_service
from src.services.qdrant_client import qdrant_service
from src.models.db import SessionLocal
from src.models.orm_models import Paper, Chunk
import uuid
import os
import shutil
import logging
from typing import List, Dict, Any
from src.config import settings
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

router = APIRouter()
logger = logging.getLogger(__name__)

UPLOAD_DIR = "uploaded_papers"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def _sanitize_filename(original_name: str) -> str:
    """
    Return a safe filename (basename only — UUID prefix added later if DB says it's duplicate).
    """
    base = os.path.basename(original_name or "")
    if not base:
        base = f"paper_{uuid.uuid4().hex}.pdf"
    return base



@router.post("/api/papers/upload")
def upload_papers(files: List[UploadFile] = File(...)):
    """
    Upload one or more PDF files, extract, chunk, embed and index into Qdrant.
    - Each file is processed atomically: if indexing fails we rollback DB changes for that file and remove the file on disk.
    - Returns list of processed files and per-file errors (if any).
    """
    processed: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for f in files:
        # Per-file handling so one bad file doesn't abort the whole batch
        safe_filename = _sanitize_filename(f.filename)
        dest = os.path.join(UPLOAD_DIR, safe_filename)

        # Save file to disk (use a temp name then rename on success to avoid partial reads)
        tmp_dest = dest + ".tmp"
        try:
            with open(tmp_dest, "wb") as out:
                shutil.copyfileobj(f.file, out)
            # atomic rename
            os.replace(tmp_dest, dest)
        except Exception as e:
            logger.exception("Failed to save uploaded file %s", f.filename)
            # cleanup tmp file if present
            try:
                if os.path.exists(tmp_dest):
                    os.remove(tmp_dest)
            except Exception:
                pass
            errors.append({"filename": f.filename, "error": "failed_to_save_file"})
            continue

        db = SessionLocal()
        try:
            # 1) Extract PDF
            try:
                info = extract_pdf(dest)
            except Exception:
                logger.exception("Failed to extract PDF for %s", dest)
                raise RuntimeError("pdf_extraction_failed")

            pages = info.get("pages") or []
            num_pages = info.get("num_pages") or len(pages)
            if not pages:
                # treat empty PDF as error
                logger.error("No pages extracted from %s", dest)
                raise RuntimeError("pdf_has_no_pages")

            # 2) Begin DB transaction for this paper + chunks (we will commit only after qdrant upsert succeeds)
            try:
                with db.begin():
                    # ensure unique filename in DB. If collision, append uuid to stored filename.
                    stored_filename = safe_filename
                    existing = db.query(Paper).filter(Paper.filename == stored_filename).first()
                    if existing:
                        # append prefix to avoid IntegrityError and preserve original name
                        stored_filename = f"{uuid.uuid4().hex}_{stored_filename}"

                    paper = Paper(
                        title=info.get("title") or f.filename,
                        authors=info.get("authors") or "",
                        year=info.get("year"),
                        filename=stored_filename,
                        num_pages=num_pages,
                    )
                    db.add(paper)
                    db.flush()  # assign paper.id so we can reference it before commit

                    # 3) Chunk
                    chunks = chunk_text_with_sections(pages)
                    if not chunks:
                        logger.warning("No chunks created for %s", dest)
                        # still proceed but note it
                        # We'll create no chunk DB entries or Qdrant items.
                    texts = [c["text"] for c in chunks]

                    # 4) Embeddings
                    emb_service = get_embedding_service()
                    try:
                        embeddings = emb_service.embed_texts(texts)
                    except Exception:
                        logger.exception("Embedding failed for %s", dest)
                        raise RuntimeError("embedding_failed")

                    if len(embeddings) != len(chunks):
                        logger.error("Embeddings count (%d) != chunks count (%d) for %s", len(embeddings), len(chunks), dest)
                        raise RuntimeError("embedding_count_mismatch")

                    # prepare qdrant items and chunk DB objects (do NOT commit yet)
                    items = []
                    for i, c in enumerate(chunks):
                        qid = str(uuid.uuid4())
                        payload = {
                            "paper_id": paper.id,
                            "paper_filename": paper.filename,
                            "paper_title": paper.title,
                            "page_start": c.get("page_start"),
                            "page_end": c.get("page_end"),
                            "section": c.get("section"),
                            "text": c.get("text"),
                        }
                        items.append({"id": qid, "vector": embeddings[i], "payload": payload})
                        chunk_db = Chunk(
                            paper_id=paper.id,
                            page_start=c.get("page_start"),
                            page_end=c.get("page_end"),
                            section=c.get("section"),
                            text=c.get("text"),
                            metadata=payload,
                            qdrant_id=qid,
                            embedding_dim=len(embeddings[i]) if embeddings[i] is not None else None,
                        )
                        db.add(chunk_db)
                    # At this point, DB transaction is still open (not committed).
                    # We'll attempt Qdrant upsert; on success the with-block will commit.
            except SQLAlchemyError:
                logger.exception("Database transaction setup failed for %s", dest)
                raise RuntimeError("db_transaction_failed")

            # 5) Upsert vectors to Qdrant (outside DB transaction to avoid long DB locks)
            try:
                if items:
                    qdrant_service.upsert_vectors(items)
            except Exception:
                logger.exception("Qdrant upsert failed for %s", dest)
                # If Qdrant fails, rollback the DB (the with db.begin() scope above would have ended;
                # to undo we remove the chunks and paper created for this file)
                try:
                    # remove created chunks + paper if present
                    db.rollback()
                    # attempt to delete any chunk rows that might have been added
                    # (safer to attempt a delete by filename)
                    db.query(Chunk).filter(Chunk.metadata['paper_filename'].astext == paper.filename).delete(synchronize_session=False)
                    db.query(Paper).filter(Paper.filename == paper.filename).delete(synchronize_session=False)
                    db.commit()
                except Exception:
                    logger.exception("Failed to cleanup DB after Qdrant failure for %s", dest)
                # remove file from disk to avoid orphaned file
                try:
                    if os.path.exists(dest):
                        os.remove(dest)
                except Exception:
                    logger.exception("Failed to remove file after Qdrant failure: %s", dest)
                errors.append({"filename": f.filename, "error": "qdrant_upsert_failed"})
                continue

            # If upsert succeeded, finalize: commit was performed within the context manager above.
            processed.append({"filename": f.filename, "paper_id": paper.id, "num_chunks": len(chunks)})
        except RuntimeError as rte:
            # Known runtime errors (we raised with short codes)
            logger.exception("Processing failed for %s: %s", dest, rte)
            # attempt clean-up and rollback
            try:
                db.rollback()
            except Exception:
                pass
            try:
                if os.path.exists(dest):
                    os.remove(dest)
            except Exception:
                logger.exception("Failed to remove file after error: %s", dest)
            errors.append({"filename": f.filename, "error": str(rte)})
            continue
        except IntegrityError as ie:
            logger.exception("Database integrity error for %s", dest)
            try:
                db.rollback()
            except Exception:
                pass
            errors.append({"filename": f.filename, "error": "integrity_error"})
            # remove file from disk
            try:
                if os.path.exists(dest):
                    os.remove(dest)
            except Exception:
                pass
            continue
        except Exception as e:
            logger.exception("Unexpected error while processing %s", dest)
            try:
                db.rollback()
            except Exception:
                pass
            try:
                if os.path.exists(dest):
                    os.remove(dest)
            except Exception:
                logger.exception("Failed to remove file after unexpected error: %s", dest)
            errors.append({"filename": f.filename, "error": "unexpected_error"})
            continue
        finally:
            try:
                db.close()
            except Exception:
                pass

    # Final response shows what succeeded and what failed (no raw exception messages)
    result = {"processed": processed}
    if errors:
        result["errors"] = errors
    return result


@router.post("/api/query")
def query(payload: dict):
    # payload: question, top_k, paper_ids (optional)
    question = payload.get("question")
    try:
        top_k = int(payload.get("top_k", 5))
    except Exception:
        top_k = 5
    paper_ids = payload.get("paper_ids")
    if not question:
        raise HTTPException(status_code=400, detail="question required")
    from src.services.rag_pipeline import run_query
    try:
        result = run_query(question, top_k=top_k, paper_ids=paper_ids)
    except Exception:
        logger.exception("run_query failed for question")
        raise HTTPException(status_code=500, detail="internal server error")
    return result



@router.get("/api/papers")
def list_papers():
    db = SessionLocal()
    try:
        papers = db.query(Paper).all()
        return [
            {
                "id": p.id,
                "title": p.title,
                "authors": p.authors,
                "year": p.year,
                "filename": p.filename,
                "num_pages": p.num_pages,
            }
            for p in papers
        ]
    finally:
        db.close()


@router.get("/api/papers/{paper_id}")
def get_paper(paper_id: int):
    db = SessionLocal()
    try:
        p = db.query(Paper).filter(Paper.id == paper_id).first()
        if not p:
            raise HTTPException(status_code=404, detail="Paper not found")
        return {"id": p.id, "title": p.title, "authors": p.authors, "year": p.year, "filename": p.filename, "num_pages": p.num_pages}
    finally:
        db.close()


@router.delete("/api/papers/{paper_id}")
def delete_paper(paper_id: int):
    db = SessionLocal()
    try:
        p = db.query(Paper).filter(Paper.id == paper_id).first()
        if not p:
            raise HTTPException(status_code=404, detail="not found")
        filename = p.filename
        # delete vectors from qdrant (catch errors but proceed to remove DB record if possible)
        try:
            qdrant_service.delete_vectors_for_paper(filename)
        except Exception:
            db.rollback()
            logger.exception("Failed to delete vectors for paper %s in Qdrant", filename)
            raise HTTPException(status_code=500, detail="failed to delete Qdrant vectors")
            # proceed to delete DB row anyway (alternatively, you could abort)
        try:
            db.delete(p)
            db.commit()
        except Exception:
            db.rollback()
            logger.exception("Failed to delete paper record %s from DB", filename)
            raise HTTPException(status_code=500, detail="failed to delete paper")
        return {"deleted": paper_id}
    finally:
        db.close()


@router.get("/api/papers/{paper_id}/stats")
def paper_stats(paper_id: int):
    db = SessionLocal()
    try:
        p = db.query(Paper).filter(Paper.id == paper_id).first()
        if not p:
            raise HTTPException(status_code=404, detail="not found")
        num_chunks = db.query(Chunk).filter(Chunk.paper_id == p.id).count()
        return {"paper_id": p.id, "num_chunks": num_chunks, "num_pages": p.num_pages}
    finally:
        db.close()


@router.get("/api/queries/history")
def query_history(limit: int = 50):
    db = SessionLocal()
    try:
        from src.models.orm_models import QueryHistory

        qs = db.query(QueryHistory).order_by(QueryHistory.created_at.desc()).limit(limit).all()
        return [
            {
                "id": q.id,
                "query_text": q.query_text,
                "papers_referenced": q.papers_referenced,
                "response_time_ms": q.response_time_ms,
                "created_at": q.created_at,
            }
            for q in qs
        ]
    finally:
        db.close()

from src.models.orm_models import QueryHistory
@router.get("/api/analytics/popular")
def analytics_popular(limit: int = 20):
    db = SessionLocal()
    from collections import Counter

    # define excluded/common terms
    excluded_terms = {
        
        # common stopwords
        "a", "an", "the", "and", "or", "is", "are", "was", "were",
        "to", "of", "in", "on", "for", "with", "as", "by", "from",
        "this", "that", "it", "at", "be", "can", "do", "how", "what",
        "why", "when", "where", "which", "who", "whom", "about", "into",
        "your", "you", "we", "they", "i","me", "has", "have", "had", "will", "would", "should", "could", "may", "might"
    }

    try:
        qs = db.query(QueryHistory).order_by(QueryHistory.created_at.desc()).limit(200).all()

        words = Counter()
        for q in qs:
            text = q.query_text if hasattr(q, "query_text") else (q[1] if len(q) > 1 else "")
            for w in (text or "").lower().split():
                word = w.strip(".,?()")
                if word and word not in excluded_terms:
                    words[word] += 1

        # only keep words with count > 3
        filtered = [(word, count) for word, count in words.items() if count > 3]

        # sort and limit results
        top = sorted(filtered, key=lambda x: x[1], reverse=True)[:limit]
        return {"top_terms": top}
    finally:
        db.close()
