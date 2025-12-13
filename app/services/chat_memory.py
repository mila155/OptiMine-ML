from typing import List, Dict
from collections import defaultdict

# Simple in-memory storage
# (bisa diganti Redis / DB nanti)
_CHAT_MEMORY = defaultdict(list)

MAX_HISTORY = 6  # biar context ga kepanjangan

def get_history(session_id: str) -> List[Dict[str, str]]:
    return _CHAT_MEMORY.get(session_id, [])[-MAX_HISTORY:]

def append_message(session_id: str, role: str, content: str):
    _CHAT_MEMORY[session_id].append({
        "role": role,
        "content": content
    })
