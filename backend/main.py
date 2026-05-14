from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from database import Base, engine, SessionLocal
from models import ChatHistory
from schemas import ChatRequest, ChatResponse, HistoryResponse
from rag import get_answer

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Portfolio Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/")
def home():
    return {"message": "Portfolio Chatbot API is running"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, db: Session = Depends(get_db)):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    old_chats = (
        db.query(ChatHistory)
        .filter(ChatHistory.session_id == request.session_id)
        .order_by(ChatHistory.created_at.desc())
        .limit(5)
        .all()
    )

    history_text = ""

    for item in reversed(old_chats):
        history_text += f"User: {item.user_message}\n"
        history_text += f"Bot: {item.bot_response}\n\n"

    answer = get_answer(
        question=request.message,
        history=history_text
    )

    chat_record = ChatHistory(
        session_id=request.session_id,
        user_message=request.message,
        bot_response=answer
    )

    db.add(chat_record)
    db.commit()

    return {"answer": answer}


@app.get("/history/{session_id}", response_model=list[HistoryResponse])
def get_history(session_id: str, db: Session = Depends(get_db)):
    chats = (
        db.query(ChatHistory)
        .filter(ChatHistory.session_id == session_id)
        .order_by(ChatHistory.created_at.asc())
        .all()
    )

    return chats