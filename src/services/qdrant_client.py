from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from src.config import settings
import logging
import math
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class QdrantService:
    def __init__(self, vector_size: Optional[int] = None, prefer_grpc: bool = False):
        url = settings.QDRANT_URL
        api_key = settings.QDRANT_API_KEY or None
        self.collection_name = settings.QDRANT_COLLECTION
        self.vector_size = vector_size or getattr(settings, "EMBEDDING_DIM", None) or 768

        try:
            self.client = QdrantClient(url, prefer_grpc=prefer_grpc, api_key=api_key)
            logger.info("Qdrant client initialized at %s", url)
        except Exception as e:
            logger.exception("Failed to initialize Qdrant client: %s", e)
            raise RuntimeError(f"Failed to initialize Qdrant client: {e}") from e

        self._ensure_collection()

    def _ensure_collection(self):
        """Create the collection only if it does not exist. Avoid destructive recreate."""
        try:
            self.client.get_collection(self.collection_name)
            logger.debug("Qdrant collection '%s' already exists.", self.collection_name)
        except Exception:
            # collection not found or call failed -> create it
            try:
                logger.info("Creating Qdrant collection '%s' with size=%s", self.collection_name, self.vector_size)
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=rest.VectorParams(size=self.vector_size, distance=rest.Distance.COSINE),
                )
            except Exception as e:
                logger.exception("Failed to create Qdrant collection '%s': %s", self.collection_name, e)
                raise RuntimeError(f"Failed to create Qdrant collection: {e}") from e

    def upsert_vectors(self, items: List[Dict[str, Any]], batch_size: int = 500):
        """
        Upsert points in batches to avoid large requests.
        items: list of {"id": str, "vector": [...], "payload": {...}}
        """
        if not items:
            return

        # basic validation
        for it in items:
            if "id" not in it or "vector" not in it:
                raise ValueError("Each item must contain 'id' and 'vector' fields")

        try:
            total = len(items)
            batches = math.ceil(total / batch_size)
            for i in range(batches):
                start = i * batch_size
                end = start + batch_size
                batch = items[start:end]
                ids = [it["id"] for it in batch]
                vectors = [it["vector"] for it in batch]
                payloads = [it.get("payload", {}) for it in batch]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=rest.Batch(ids=ids, vectors=vectors, payloads=payloads),
                )
        except Exception as e:
            logger.exception("Failed to upsert vectors: %s", e)
            raise RuntimeError(f"Qdrant upsert failed: {e}") from e

    def search(self, vector: List[float], top_k: int = 5, filter: Optional[rest.Filter] = None, with_payload: bool = True):
        try:
            hits = self.client.search(
                collection_name=self.collection_name,
                query_vector=vector,
                limit=top_k,
                with_payload=with_payload,
                query_filter=filter,
            )
            return hits
        except Exception as e:
            logger.exception("Qdrant search failed: %s", e)
            raise RuntimeError(f"Qdrant search failed: {e}") from e

    def delete_vectors_for_paper(self, paper_filename: str):
        """Delete points whose payload 'paper_filename' equals the given filename."""
        try:
            q_filter = rest.Filter(must=[rest.FieldCondition(key="paper_filename", match=rest.MatchValue(value=paper_filename))])
            self.client.delete(collection_name=self.collection_name, filter=q_filter)
        except Exception as e:
            logger.exception("Failed to delete vectors for paper '%s': %s", paper_filename, e)
            raise RuntimeError(f"Qdrant delete failed: {e}") from e

# create per-process instance (avoid call at import time in some apps if you prefer lazy creation)
qdrant_service = QdrantService()
