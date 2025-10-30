from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    JSON,
    ForeignKey,
    Float,
)
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from src.models.db import Base


class Paper(Base):
    __tablename__ = "papers"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False, index=True)
    authors = Column(String, nullable=False)
    year = Column(Integer, nullable=True)
    filename = Column(String, unique=True, nullable=False)
    num_pages = Column(Integer, nullable=True)
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationship — a Paper can have many Chunks
    chunks = relationship(
        "Chunk",
        back_populates="paper",
        cascade="all, delete-orphan",
        passive_deletes=True
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, index=True)
    paper_id = Column(Integer, ForeignKey("papers.id", ondelete="CASCADE"), nullable=False)
    page_start = Column(Integer, nullable=True)
    page_end = Column(Integer, nullable=True)
    section = Column(String, nullable=True)
    text = Column(Text, nullable=False)
    meta = Column("metadata", JSON, nullable=True) 
    qdrant_id = Column(String, index=True, nullable=True)  # vector id in Qdrant
    embedding_dim = Column(Integer, nullable=True)

    # Relationship — a Chunk belongs to one Paper
    paper = relationship("Paper", back_populates="chunks")


class QueryHistory(Base):
    __tablename__ = "query_history"

    id = Column(Integer, primary_key=True, index=True)
    query_text = Column(Text, nullable=False)
    papers_referenced = Column(JSON, nullable=True)
    response_time_ms = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    result_summary = Column(Text, nullable=True)
