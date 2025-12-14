# app/services/chatbot_service.py
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

from app.services.chat_context_builder import build_context
from app.rag.rag_engine import RAGEngineSafe 
from app.services.llm_service import LLMService

class ChatbotService:
    def __init__(self, rag_engine: Optional[Any] = None):
        self.llm_service = LLMService()
        self.rag_engine = rag_engine
        self.conversation_history: Dict[str, List[Dict]] = {}
        self.role_instructions = {
            'mining_planner': """Anda adalah PERENCANA OPERASIONAL PERTAMBANGAN di perusahaan OptiMine.

Tugas Utama Anda:
1. Perencanaan produksi harian/bulanan
2. Optimasi alokasi alat berat (excavator, dump truck)
3. Scheduling maintenance equipment
4. Monitoring KPI produksi
5. Analisis gap produksi vs target

Fokus pada:
- Efisiensi operasional
- Cost optimization
- Risk mitigation
- Compliance dengan SOP""",

            'shipping_planner': """Anda adalah PERENCANA PENGAPALAN di perusahaan OptiMine.

Tugas Utama Anda:
1. Perencanaan jadwal kapal (vessel scheduling)
2. Optimasi loading/unloading sequence
3. Koordinasi hauling ROM-to-Jetty
4. Monitoring demurrage cost
5. Analisis bottleneck di jetty

Fokus pada:
- Minimasi waktu tunggu kapal
- Optimasi utilisasi jetty
- Reduksi biaya demurrage
- Koordinasi dengan mining planner""",

            'general': "Anda adalah asisten AI untuk perusahaan pertambangan dan pengapalan."
        }
        
    def handle_chat(
        self,
        session_id: str,
        message: str,
        context_type: Optional[str] = None,
        context_payload: Optional[Dict[str, Any]] = None,
        role: str = "general"
    ) -> Dict[str, Any]:
        """
        Handle chat message dengan hanya 2 role: mining_planner dan shipping_planner
        """
        try:
            # Validasi role
            if role not in ['mining_planner', 'shipping_planner', 'general']:
                role = 'general'  # Fallback ke general
            
            # Validasi input
            if not message or not session_id:
                return self._error_response("Pesan dan session_id diperlukan")
            
            # Inisialisasi session
            if session_id not in self.conversation_history:
                self.conversation_history[session_id] = []
            
            # 1. Get RAG context untuk dokumen pengetahuan
            rag_context = self._get_rag_context(message)
            
            # 2. Get structured context dari frontend
            structured_context = self._get_structured_context(context_type, context_payload)
            
            # 3. Dapatkan instruction berdasarkan role
            role_instruction = self.role_instructions.get(role, self.role_instructions['general'])
            
            # 4. Get conversation history
            history_text = self._get_history_text(session_id)
            
            # 5. Build system prompt khusus berdasarkan role
            system_prompt = self._build_role_specific_prompt(
                role_instruction=role_instruction,
                message=message,
                structured_context=structured_context,
                rag_context=rag_context,
                history_text=history_text,
                context_type=context_type
            )
            
            # 6. Generate response
            answer = self._generate_response(system_prompt)
            
            # 7. Extract sources
            sources = self._extract_sources(rag_context)
            
            # 8. Update conversation history
            self._update_history(session_id, message, answer, context_type, role)
            
            # 9. Return response
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
            
        except Exception as e:
            print(f"❌ Error in handle_chat: {e}")
            return self._error_response(f"Terjadi kesalahan sistem: {str(e)}")
    
    def _get_rag_context(self, message: str) -> str:
        """Get context from RAG documents"""
        if not self.rag_engine:
            return ""
        
        try:
            # Mining-related queries
            mining_keywords = ['excavator', 'produksi', 'tambang', 'pit', 'hauling', 'alat berat']
            # Shipping-related queries
            shipping_keywords = ['kapal', 'jetty', 'loading', 'demurrage', 'shipping', 'vessel']
            
            # Adjust k based on query type
            k = 5
            if any(keyword in message.lower() for keyword in mining_keywords):
                k = 4  # Lebih fokus untuk mining
            elif any(keyword in message.lower() for keyword in shipping_keywords):
                k = 4  # Lebih fokus untuk shipping
            
            context = self.rag_engine.get_context(message, k=k)
            return context
        except Exception as e:
            print(f"⚠️ RAG error: {e}")
            return ""
    
    def _get_structured_context(self, context_type: str, context_payload: Dict) -> str:
        """Get structured context from payload"""
        if not context_type or not context_payload:
            return ""
        
        try:
            return build_context(context_type, context_payload)
        except Exception as e:
            print(f"⚠️ Context building error: {e}")
            return ""
    
    def _get_history_text(self, session_id: str) -> str:
        """Get conversation history"""
        history = self.conversation_history.get(session_id, [])
        if not history:
            return "Tidak ada riwayat percakapan sebelumnya"
        
        history_text = "RIWAYAT PERCAKAPAN:\n"
        for i, h in enumerate(history[-3:], 1):  # Last 3 messages
            history_text += f"{i}. User: {h['question']}\n"
            history_text += f"   Assistant: {h['answer'][:100]}...\n"
        
        return history_text
    
    def _build_role_specific_prompt(
        self,
        role_instruction: str,
        message: str,
        structured_context: str,
        rag_context: str,
        history_text: str,
        context_type: str
    ) -> str:
        """Build prompt specific to role"""
        
        # Tentukan format output berdasarkan context_type
        output_guidance = ""
        if context_type == 'mining_summary':
            output_guidance = """
            FORMAT JAWABAN (untuk mining summary):
            1. Analisis singkat data
            2. Identifikasi 3 masalah utama
            3. Rekomendasi prioritas untuk planner
            4. Action items spesifik
            """
        elif context_type == 'shipping_summary':
            output_guidance = """
            FORMAT JAWABAN (untuk shipping summary):
            1. Analisis bottleneck
            2. Rekomendasi scheduling
            3. Optimasi jetty utilization
            4. Action plan untuk reduce demurrage
            """
        
        prompt = f"""{role_instruction}

{history_text}

{'='*60}
KONTEKS DATA SISTEM:
{structured_context if structured_context else 'Tidak ada data spesifik dari sistem'}
{'='*60}

{'='*60}
REFERENSI DOKUMEN:
{rag_context if rag_context else 'Tidak ada referensi dokumen yang relevan'}
{'='*60}

{output_guidance}

PERTANYAAN: {message}

PETUNJUK JAWABAN:
1. Gunakan bahasa Indonesia profesional
2. Fokus pada aspek perencanaan dan optimasi
3. Berikan jawaban yang actionable
4. Jika data tidak cukup, minta klarifikasi

JAWABAN:"""
        
        return prompt
    
    def _generate_response(self, prompt: str) -> str:
        """Generate response from LLM"""
        try:
            return self.llm_service.generate(
                prompt=prompt,
                max_tokens=1000,
                temperature=0.7
            )
        except Exception as e:
            print(f"❌ LLM error: {e}")
            return "Maaf, saya tidak dapat memberikan jawaban saat ini. Silakan coba lagi."
    
    def _update_history(self, session_id: str, message: str, answer: str, context_type: str, role: str):
        """Update conversation history"""
        self.conversation_history[session_id].append({
            "timestamp": datetime.now().isoformat(),
            "question": message,
            "answer": answer,
            "context_type": context_type,
            "role": role
        })
        
        # Trim history
        if len(self.conversation_history[session_id]) > 10:
            self.conversation_history[session_id] = self.conversation_history[session_id][-10:]
    
    def _extract_sources(self, rag_context: str) -> List[str]:
        """Extract sources from RAG context"""
        if not rag_context:
            return []
        
        sources = []
        lines = rag_context.split('\n')
        for line in lines:
            if '[' in line and ']' in line:
                start = line.find('[')
                end = line.find(']')
                if start != -1 and end != -1:
                    sources.append(line[start:end+1])
        
        return list(set(sources))[:3]
    
    def _error_response(self, error_msg: str) -> Dict[str, Any]:
        """Create error response"""
        return {
            "answer": f"⚠️ {error_msg}",
            "sources": [],
            "timestamp": datetime.now().isoformat(),
            "context_used": {}
        }