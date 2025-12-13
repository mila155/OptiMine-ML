from typing import Dict, Any, List
from datetime import datetime

from app.services.chat_memory import get_history, append_message
from app.services.chat_context_builder import build_context
from app.services.llm_service import LLMService
from app.rag.rag_engine import RAGEngineSafe



class ChatbotService:
    """
    Core service untuk chatbot OptiMine
    - Session memory
    - Context injection (summary / optimization / selected plan)
    - RAG (dokumen)
    """

    def __init__(self):
        self.llm = LLMService()
        self.rag = RAGEngineSafe()

    def handle_chat(
        self,
        session_id: str,
        message: str,
        context_type: str = None,
        context_payload: Dict[str, Any] = None,
        role: str = None
    ) -> Dict[str, Any]:

        # ========================
        # 1. Load chat history
        # ========================
        history = get_history(session_id)

        # ========================
        # 2. Build injected context
        # ========================
        structured_context = build_context(context_type, context_payload)

        # ========================
        # 3. RAG retrieval
        # ========================
        rag_sources = []
        rag_context = ""

        if message:
            docs = self.rag.get_context(message)
            if docs:
                for d in docs:
                    rag_sources.append({
                        "id": d.get("id"),
                        "score": d.get("score"),
                        "metadata": d.get("metadata")
                    })
                    rag_context += f"\n[SOURCE {d.get('id')}] {d.get('text')[:800]}"

        # ========================
        # 4. System prompt
        # ========================
        system_prompt = f"""
You are OptiMine AI Assistant.

ROLE CONTEXT:
User role: {role or "General Planner"}

You are expert in:
- Mining production planning
- Hauling & ROM operations
- Shipping & jetty scheduling
- Optimization & scenario analysis

RULES:
- Use provided context & documents
- Be concise, structured, professional
- Do not hallucinate numbers
- Explain reasoning clearly

STRUCTURED CONTEXT:
{structured_context}

REFERENCE DOCUMENTS:
{rag_context}
"""

        # ========================
        # 5. Compose messages
        # ========================
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]

        for h in history:
            messages.append(h)

        messages.append({
            "role": "user",
            "content": message
        })

        # ========================
        # 6. Call LLM
        # ========================
        ai_answer = self.llm.chat(messages)

        # ========================
        # 7. Save memory
        # ========================
        append_message(session_id, "user", message)
        append_message(session_id, "assistant", ai_answer)

        # ========================
        # 8. Response
        # ========================
        return {
            "answer": ai_answer,
            "sources": rag_sources if rag_sources else None,
            "timestamp": datetime.utcnow()
        }
