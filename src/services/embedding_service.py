from sentence_transformers import SentenceTransformer
from src.config import settings
import numpy as np
import logging
import threading
from typing import List, Optional

logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    Wrapper around SentenceTransformer with error handling, batching, and optional normalization.
    """

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        logger.info("Loading embedding model %s ...", self.model_name)
        try:
            self.model = SentenceTransformer(self.model_name)
            # optional: move model to specified device (e.g., "cuda" or "cpu")
            if device:
                try:
                    self.model.to(device)
                    logger.info("Moved embedding model to device: %s", device)
                except Exception as e:
                    logger.warning("Failed to move model to device %s: %s", device, e)
        except Exception as e:
            logger.exception("Failed to load SentenceTransformer model '%s': %s", self.model_name, e)
            # raise a clear error so the caller can handle startup failure
            raise RuntimeError(f"Failed to load embedding model '{self.model_name}': {e}") from e

    def embed_texts(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        normalize: bool = False,
        show_progress_bar: bool = False,
    ) -> List[List[float]]:
        """
        Embed a list of texts and return list-of-list floats.

        Args:
          texts: list of strings
          batch_size: optional int to control memory usage (passed to model.encode)
          normalize: if True, L2-normalize vectors (useful for cosine similarity)
          show_progress_bar: set to True for local debugging; keep False in servers.

        Raises:
          ValueError on invalid input.
          RuntimeError on model encoding failures.
        """
        # Input validation
        if texts is None:
            raise ValueError("texts must be a list of strings, got None")
        if not isinstance(texts, list):
            raise ValueError(f"texts must be a list of strings, got {type(texts)}")
        if not texts:
            return []

        for i, t in enumerate(texts):
            if not isinstance(t, str):
                raise ValueError(f"All items in texts must be str. Index {i} is {type(t)}")

        try:
            encode_kwargs = {"show_progress_bar": show_progress_bar, "convert_to_numpy": True}
            if batch_size is not None:
                encode_kwargs["batch_size"] = int(batch_size)

            arr = self.model.encode(texts, **encode_kwargs)  # numpy array expected
            if not isinstance(arr, np.ndarray):
                # ensure we have a numpy array for downstream numeric ops
                arr = np.array(arr)

            if normalize:
                # L2 normalize per-row; guard against zero vectors
                norms = np.linalg.norm(arr, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                arr = arr / norms

            # convert to Python lists (JSON-serializable)
            return arr.astype(float).tolist()

        except Exception as e:
            logger.exception("Failed to encode texts (model: %s): %s", self.model_name, e)
            # raise a runtime error so upper layers (FastAPI) can convert to HTTPException
            raise RuntimeError(f"Embedding failed: {e}") from e


# Thread-safe global singleton instance to avoid repeated model loading
_embedding_service: Optional[EmbeddingService] = None
_embedding_lock = threading.Lock()

def get_embedding_service(model_name: Optional[str] = None, device: Optional[str] = None) -> EmbeddingService:
    """
    Return a process-global EmbeddingService instance.
    Optional model_name/device override for the first creation.
    Thread-safe.
    """
    global _embedding_service
    if _embedding_service is None:
        with _embedding_lock:
            if _embedding_service is None:
                _embedding_service = EmbeddingService(model_name=model_name, device=device)
    return _embedding_service
