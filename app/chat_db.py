import uuid

CHAT_SESSIONS = {}  # user -> list of chat_ids
CHAT_HISTORY = {}   # chat_id -> list of messages

def new_chat_session(user: str) -> str:
    chat_id = str(uuid.uuid4())
    if user not in CHAT_SESSIONS:
        CHAT_SESSIONS[user] = []
    CHAT_SESSIONS[user].append(chat_id)
    CHAT_HISTORY[chat_id] = []
    return chat_id

def save_message(chat_id: str, user: str, message: str, ai_response: str):
    if chat_id not in CHAT_HISTORY:
        CHAT_HISTORY[chat_id] = []
    CHAT_HISTORY[chat_id].append({"sender": "user", "text": message})
    CHAT_HISTORY[chat_id].append({"sender": "ai", "text": ai_response})

def get_chat_history(chat_id: str):
    return CHAT_HISTORY.get(chat_id, [])

def get_user_chats(user: str):
    chat_ids = CHAT_SESSIONS.get(user, [])
    result = []
    for cid in chat_ids:
        last_msg = CHAT_HISTORY[cid][-1]["text"] if CHAT_HISTORY[cid] else "New chat"
        result.append({"chat_id": cid, "last": last_msg})
    return result
