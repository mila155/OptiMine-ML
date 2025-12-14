import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

# Import RAG engine
from app.rag.rag_engine import build_context
from app.rag.rag_engine import RAGEngineSafe  # atau SafeRAGEngine dari main
from app.services.llm_service import LLMService

class ChatbotService:
    def __init__(self, rag_engine: Optional[Any] = None):
        """
        Initialize Chatbot Service
        Args:
            rag_engine: Instance of RAGEngineSafe/SafeRAGEngine (optional)
        """
        self.llm_service = LLMService()
        self.rag_engine = rag_engine
        self.conversation_history: Dict[str, List[Dict]] = {}
        
    def handle_chat(
        self,
        session_id: str,
        message: str,
        context_type: Optional[str] = None,
        context_payload: Optional[Dict[str, Any]] = None,
        role: str = "general"
    ) -> Dict[str, Any]:
        """
        Handle chat message with optional context and RAG
        """
        # 1. Validasi input
        if not message or not session_id:
            return self._error_response("Pesan dan session_id diperlukan")
        
        # 2. Inisialisasi session jika belum ada
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = []
        
        # 3. Ambil konteks dari RAG jika tersedia
        rag_context = ""
        if self.rag_engine and hasattr(self.rag_engine, 'get_context'):
            try:
                rag_context = self.rag_engine.get_context(message, k=3)
            except Exception as e:
                print(f"⚠️ RAG retrieval error: {e}")
                rag_context = ""
        
        # 4. Build structured context dari context_type
        structured_context = ""
        if context_type and context_payload:
            structured_context = build_context(context_type, context_payload)
        
        # 5. Gabungkan semua konteks
        full_context = f"""
ROLE: {role.upper()}
USER QUESTION: {message}

{'='*50}
STRUCTURED CONTEXT (jika ada):
{structured_context}
{'='*50}

DOCUMENT CONTEXT (dari RAG):
{rag_context if rag_context else 'Tidak ada dokumen referensi tersedia'}
{'='*50}
"""
        
        # 6. Tambahkan conversation history (last 3 messages)
        history = self.conversation_history[session_id][-3:] if self.conversation_history[session_id] else []
        history_text = "\n".join([
            f"User: {h['question']}\nAssistant: {h['answer'][:100]}..." 
            for h in history
        ]) if history else "Tidak ada riwayat percakapan sebelumnya"
        
        # 7. Format prompt untuk LLM
        system_prompt = f"""Anda adalah asisten AI untuk perusahaan pertambangan dan pengapalan (OptiMine).
Peran Anda: {role}
        
Panduan:
1. Jawab dalam bahasa Indonesia yang profesional
2. Gunakan konteks yang diberikan (jika ada)
3. Jika tidak yakin, katakan "Saya tidak memiliki informasi cukup tentang itu"
4. Berikan jawaban yang jelas dan terstruktur
5. Fokus pada optimasi operasional

Riwayat Percakapan:
{history_text}

Konteks Lengkap:
{full_context}

Pertanyaan User: {message}

Jawaban:"""
        
        # 8. Generate response dari LLM
        try:
            answer = self.llm_service.generate(
                prompt=system_prompt,
                max_tokens=1000,
                temperature=0.7
            )
        except Exception as e:
            print(f"LLM error: {e}")
            answer = f"Maaf, terjadi kesalahan dalam memproses pertanyaan Anda. Error: {str(e)}"
        
        # 9. Extract sources dari RAG context (jika ada)
        sources = self._extract_sources(rag_context)
        
        # 10. Update conversation history
        self.conversation_history[session_id].append({
            "timestamp": datetime.now().isoformat(),
            "question": message,
            "answer": answer,
            "context_type": context_type,
            "role": role
        })
        
        # 11. Trim history (max 10 messages per session)
        if len(self.conversation_history[session_id]) > 10:
            self.conversation_history[session_id] = self.conversation_history[session_id][-10:]
        
        # 12. Return response
        return {
            "answer": answer,
            "sources": sources,
            "timestamp": datetime.now().isoformat(),
            "context_used": {
                "has_structured_context": bool(structured_context),
                "has_rag_context": bool(rag_context),
                "role": role
            }
        }
    
    def _extract_sources(self, rag_context: str) -> List[str]:
        """Extract sources from RAG context string"""
        if not rag_context:
            return []
        
        sources = []
        lines = rag_context.split('\n')
        for line in lines:
            if 'Sumber:' in line or '[Sumber:' in line:
                # Extract source info
                import re
                source_match = re.search(r'\[?(Sumber:\s*[^\]]+)\]?', line)
                if source_match:
                    sources.append(source_match.group(1))
        
        return sources[:3]  # Max 3 sources
    
    def _error_response(self, error_msg: str) -> Dict[str, Any]:
        """Create error response"""
        return {
            "answer": f"⚠️ {error_msg}",
            "sources": [],
            "timestamp": datetime.now().isoformat(),
            "context_used": {}
        }
    
    def clear_history(self, session_id: str) -> bool:
        """Clear conversation history for a session"""
        if session_id in self.conversation_history:
            del self.conversation_history[session_id]
            return True
        return False