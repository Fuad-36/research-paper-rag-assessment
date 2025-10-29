from pydantic import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str
    QDRANT_URL: str
    QDRANT_API_KEY: str | None = None
    EMBEDDING_MODEL: str = "all-mpnet-base-v2"
    LLM_PROVIDER: str = "ollama"
    OLLAMA_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3:latest"
    MAX_CHUNK_CHARS: int = 1800
    CHUNK_OVERLAP_CHARS: int = 200
    QDRANT_COLLECTION: str = "papers"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000

    class Config:
        env_file = ".env"

settings = Settings()
