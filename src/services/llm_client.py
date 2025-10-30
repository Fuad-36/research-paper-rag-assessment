import json
import requests
import logging
from typing import Any, Dict, Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from src.config import settings

logger = logging.getLogger(__name__)


class OllamaError(RuntimeError):
    """Raised for Ollama client errors (network, HTTP, parsing)."""
    pass


class OllamaClient:
    """
    Robust synchronous client for an Ollama-like HTTP endpoint.

    - Retries idempotent failures with exponential backoff.
    - Validates inputs.
    - Handles non-JSON responses gracefully.
    - Does NOT add authentication code (per request).
    """

    def __init__(
        self,
        base_url: str = settings.OLLAMA_URL,
        model: str = settings.OLLAMA_MODEL,
        timeout: int = 60,
        max_retries: int = 3,
        backoff_factor: float = 0.3,
    ):
        self.base_url = (base_url or "").rstrip("/")
        self.model = model
        self.timeout = int(timeout)

        # Prepare a requests.Session with retries for transient errors
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(["POST", "GET", "HEAD", "OPTIONS"]),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        # Mount adapter for http and https
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        logger.info("OllamaClient initialized for model=%s at %s", self.model, self.base_url)

    def _validate_params(self, prompt: str, max_tokens: int, temperature: float) -> None:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if not isinstance(temperature, (int, float)) or temperature < 0:
            raise ValueError("temperature must be a non-negative number")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        timeout: Optional[int] = None,
        stream: Optional[bool] = None,  # optional override
    ) -> Dict[str, Any]:
        self._validate_params(prompt, max_tokens, temperature)
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        # Let user explicitly disable/enable streaming; default = server default
        if stream is not None:
            payload["stream"] = bool(stream)

        req_timeout = timeout or self.timeout

        try:
            # request with streaming so we can parse NDJSON incrementally
            resp = self.session.post(url, json=payload, timeout=req_timeout, stream=True)
        except requests.exceptions.RequestException as e:
            logger.exception("Network error when calling Ollama at %s: %s", url, e)
            raise OllamaError(f"Network error calling LLM: {e}") from e

        if not resp.ok:
            # try to show useful body for debugging
            try:
                body = resp.json()
            except Exception:
                body = resp.text[:1000]
            logger.error("Ollama returned HTTP %s: %s", resp.status_code, body)
            raise OllamaError(f"Ollama returned status {resp.status_code}: {body}")

        # We'll try streaming NDJSON first (the usual Ollama behavior).
        chunks = []
        try:
            # iter_lines(decode_unicode=True) returns each line (NDJSON)
            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                # Sometimes lines may be "data: ..." or other prefixes; handle common prefixes:
                line = raw_line
                if isinstance(line, bytes):
                    try:
                        line = line.decode("utf-8", errors="replace")
                    except Exception:
                        line = raw_line

                # strip "data: " prefix if present (SSE-like)
                if line.startswith("data: "):
                    line = line[len("data: "):]

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    # not a JSON line — keep for debugging but skip
                    logger.debug("Non-JSON stream line from Ollama: %r", line[:200])
                    continue

                # Append textual fragment if present
                fragment = obj.get("response") or obj.get("text") or obj.get("output")
                if fragment is not None:
                    # fragment will already have escape sequences decoded by json.loads
                    chunks.append(str(fragment))

                # When the provider signals completion, break
                if obj.get("done") is True:
                    logger.debug("Ollama signalled done.")
                    break

            # join fragments into final text
            joined = "".join(chunks).strip()

            # Attempt to parse the joined text as JSON (handles nested JSON responses)
            if joined:
                # If it looks like JSON, try parsing
                try:
                    parsed = json.loads(joined)
                    return {"status_code": resp.status_code, "json": parsed, "raw_text": joined}
                except json.JSONDecodeError:
                    # not JSON — return plain text
                    return {"status_code": resp.status_code, "text": joined, "raw_chunks": chunks}
            else:
                # stream produced no 'response' fragments; fallback to returning raw body
                body = resp.text
                try:
                    data = json.loads(body)
                    return {"status_code": resp.status_code, "json": data}
                except Exception:
                    return {"status_code": resp.status_code, "raw_text": body}
        finally:
            # ensure response is closed to release connection
            try:
                resp.close()
            except Exception:
                pass


def get_llm_client() -> OllamaClient:
    """
    Factory returning a new OllamaClient instance.
    (You can wrap this with your own singleton if you want a single client per process.)
    """
    return OllamaClient()
