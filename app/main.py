from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from chat_db import save_message, get_chat_history, new_chat_session, get_user_chats
from chat_logic import get_ai_response

app = FastAPI(title="Lyra Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class ChatRequest(BaseModel):
    chat_id: str
    user: str
    message: str

class NewChatRequest(BaseModel):
    user: str

class HistoryRequest(BaseModel):
    chat_id: str

@app.get("/")
def root():
    return {"status": "ok", "message": "Lyra Chat API Running"}

@app.post("/new_chat/")
def new_chat(request: NewChatRequest):
    chat_id = new_chat_session(request.user)
    return {"chat_id": chat_id, "message": "New chat started."}

@app.post("/chat/")
def chat_endpoint(request: ChatRequest):
    if not (request.user and request.message and request.chat_id):
        raise HTTPException(status_code=400, detail="Missing data")

    # Generate AI response using LangChain Groq
    history = get_chat_history(request.chat_id)
    ai_response = get_ai_response(request.message, request.user, mood="", history=history)
    save_message(request.chat_id, request.user, request.message, ai_response)
    return {"chat_id": request.chat_id, "ai_response": ai_response}

@app.post("/history/")
def history_endpoint(request: HistoryRequest):
    history = get_chat_history(request.chat_id)
    return {"chat_id": request.chat_id, "history": history}

@app.post("/user_chats/")
def user_chats(request: NewChatRequest):
    chats = get_user_chats(request.user)
    return {"chats": chats}
