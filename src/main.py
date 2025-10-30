import uvicorn
from fastapi import FastAPI
from src.api.routes import router
import logging
from src.models.db import Base, engine
from src.models.orm_models import *
from src.config import settings

logging.basicConfig(level=logging.INFO)
app = FastAPI(title="RAG Research Assistant")
app.include_router(router)


@app.get("/")
def root():
    return {"message": "Research Paper RAG server is running 🚀"}
# ensure DB tables exist at startup
@app.on_event("startup")
def startup():
    Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    uvicorn.run("src.main:app", host=settings.APP_HOST, port=settings.APP_PORT, reload=True)
