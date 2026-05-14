from datetime import datetime
from pydantic import BaseModel


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    answer: str


class HistoryResponse(BaseModel):
    user_message: str
    bot_response: str
    created_at: datetime