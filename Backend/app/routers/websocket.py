from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from typing import Dict, Any
import json
import asyncio
import logging
from datetime import datetime
from sqlalchemy.orm import Session

from app.database import get_db
from app.services.llm import LLMService
_llm_service = LLMService()

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ws", tags=["websocket"])

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected")
    
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Client {client_id} disconnected")
    
    async def send_json(self, client_id: str, data: dict):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json(data)
    
    async def broadcast(self, data: dict):
        for connection in self.active_connections.values():
            await connection.send_json(data)

manager = ConnectionManager()

@router.websocket("/client/{client_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    client_id: str,
    db: Session = Depends(get_db)
):
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            message_type = data.get("type")
            
            if message_type == "ping":
                # Keep alive
                await manager.send_json(client_id, {"type": "pong"})
            
            elif message_type == "processing_status":
                # Document processing updates will be sent via document_processor
                pass
            
            elif message_type == "stream_answer":
                # Stream answer generation
                await stream_answer_generation(
                    client_id=client_id,
                    question=data.get("question"),
                    context_chunks=data.get("context_chunks", []),
                    conversation_context=data.get("conversation_context", []),
                    session_id=data.get("session_id", None)
                )
            
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(client_id)

async def stream_answer_generation(
    client_id: str, 
    question: str, 
    context_chunks: list,
    conversation_context: list,
    session_id: int = None
):
    """Stream answer generation token by token"""
    try:
        import time
        start_time = time.time()
        # Send start signal
        await manager.send_json(client_id, {
            "type": "answer_stream_start",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Build context-aware prompt
        context_prompt = ""
        if conversation_context:
            context_prompt = "Previous conversation:\n"
            for i, qa in enumerate(conversation_context[-3:]):
                context_prompt += f"Q{i+1}: {qa['question']}\n"
                context_prompt += f"A{i+1}: {qa['answer']}\n\n"
            context_prompt += "Current question:\n"
        
        enhanced_question = f"{context_prompt}{question}" if context_prompt else question
        # If no real context chunks, use a friendly fallback
        if not context_chunks:
            context_chunks = [{
                'text': 'The user is having a casual conversation. Respond naturally and friendly.',
                'source': 'system',
                'similarity_score': 1.0
            }]
        
        # Simulate streaming (in real implementation, modify LLM service to support streaming)
        # For now, we'll simulate by sending words progressively
        llm_service = _llm_service

        # Get full answer first
        result = llm_service.generate_answer(
            context_chunks,
            enhanced_question,
            max_tokens=512,
            temperature=0.3
        )
        
        full_answer = result.get('answer', '')
        
       # Post-process: fix inline numbered points to proper line breaks
        import re
        full_answer = re.sub(r'\s*\(i\)\s*', '\n\n• ', full_answer)
        full_answer = re.sub(r'\s*\(ii\)\s*', '\n\n• ', full_answer)
        full_answer = re.sub(r'\s*\(iii\)\s*', '\n\n• ', full_answer)
        full_answer = re.sub(r'\s*\(iv\)\s*', '\n\n• ', full_answer)
        full_answer = re.sub(r'\s*\(v\)\s*', '\n\n• ', full_answer)
        # Fix inline 1. 2. 3. patterns
        full_answer = re.sub(r'\s+(\d+)\.\s+\*\*', r'\n\n\1. **', full_answer)
        full_answer = re.sub(r'\s+(\d+)\.\s+([A-Z])', r'\n\n\1. \2', full_answer)
        # Remove extra "Hey, I'd be happy to..." intro sentences
        full_answer = re.sub(r"^Hey,?\s+I['']d be happy to [^.]+\.\s*", '', full_answer)
        full_answer = re.sub(r"^Hey,?\s+[^.]+\.\s*", '', full_answer)
        full_answer = full_answer.strip()
        
        words = full_answer.split()
        
        # Stream words with natural pacing
        current_text = ""
        for i, word in enumerate(words):
            current_text += word + " "
            
            await manager.send_json(client_id, {
                "type": "answer_stream_chunk",
                "content": current_text,
                "is_complete": False
            })
            
            # Natural typing delay - inside the loop!
            await asyncio.sleep(0.05)
        
        # ✅ Save to DB FIRST so we have the real session_id to send back
        try:
            from app.database import SessionLocal
            from app.models.query import QueryHistoryDB, AnalyticsStatsDB
            import hashlib

            db = SessionLocal()
            
            normalized = question.lower().strip().replace('?','').replace('.','')
            similarity_hash = hashlib.md5(normalized.encode()).hexdigest()

            from app.models.query import ChatSessionDB

            # Use existing session or create new one
            if session_id:
                existing_session = db.query(ChatSessionDB).filter(
                    ChatSessionDB.id == session_id
                ).first()
                if not existing_session:
                    session_id = None

            if not session_id:
                title = question[:60] + '...' if len(question) > 60 else question
                new_session = ChatSessionDB(title=title)
                db.add(new_session)
                db.flush()
                session_id = new_session.id

            query_history = QueryHistoryDB(
                session_id=session_id,
                question=question,
                answer=current_text.strip(),
                sources_count=len(context_chunks),
                response_time=round(time.time() - start_time, 2),
                llm_used=result.get('llm_used'),
                context_chunks_count=len(context_chunks),
                success="true",
                similarity_hash=similarity_hash
            )
            db.add(query_history)
            
            # Update analytics stats
            stats = db.query(AnalyticsStatsDB).first()
            if stats:
                stats.total_queries += 1

            db.commit()
            db.close()
        except Exception as db_error:
            logger.error(f"Failed to save streaming query to DB: {db_error}")

        # Send completion signal AFTER DB save so session_id is real
        await manager.send_json(client_id, {
            "type": "answer_stream_end",
            "content": current_text.strip(),
            "is_complete": True,
            "llm_used": result.get('llm_used'),
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": session_id
        })

    except Exception as e:
        logger.error(f"Error streaming answer: {e}")
        await manager.send_json(client_id, {
            "type": "answer_stream_error",
            "error": str(e)
        })

# Export for use in other modules
websocket_manager = manager