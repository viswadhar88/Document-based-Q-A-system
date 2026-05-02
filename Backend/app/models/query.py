from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict
from sqlalchemy import Column, DateTime, Integer, String, Text, Float, ForeignKey
from app.database import Base

class ChatSessionDB(Base):
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class QueryHistoryDB(Base):
    __tablename__ = "query_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"), nullable=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=True)
    sources_count = Column(Integer, default=0)
    response_time = Column(Float, default=0.0)
    llm_used = Column(String, nullable=True)
    context_chunks_count = Column(Integer, default=0)
    success = Column(String, default="true")
    created_at = Column(DateTime, default=datetime.utcnow)
    similarity_hash = Column(String, nullable=True)

class AnalyticsStatsDB(Base):
    __tablename__ = "analytics_stats"

    id = Column(Integer, primary_key=True, autoincrement=True)
    total_queries = Column(Integer, default=0)
    total_documents = Column(Integer, default=0)
    avg_response_time = Column(Float, default=0.0)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class QueryHistoryResponse(BaseModel):
    id: int
    session_id: Optional[int] = None
    question: str
    answer: Optional[str] = None
    sources_count: int
    response_time: float
    llm_used: Optional[str] = None
    context_chunks_count: int
    success: str
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)

class QueryHistoryList(BaseModel):
    queries: List[QueryHistoryResponse]
    count: int
    page: int
    limit: int

class ChatSessionResponse(BaseModel):
    id: int
    title: str
    created_at: datetime
    updated_at: datetime
    messages: Optional[List[QueryHistoryResponse]] = []
    model_config = ConfigDict(from_attributes=True)

class AnalyticsStats(BaseModel):
    total_queries: int
    total_documents: int
    avg_response_time: float
    successful_queries: int
    failed_queries: int
    last_updated: datetime
    top_llm_used: Optional[str] = None

class PopularQuestion(BaseModel):
    question: str
    frequency: int
    avg_response_time: float
    success_rate: float
    last_asked: datetime

class PopularQuestionsResponse(BaseModel):
    questions: List[PopularQuestion]
    total_unique_questions: int