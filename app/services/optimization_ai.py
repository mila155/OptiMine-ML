"""
AI helper functions for optimization plan enrichment
Handles:
- strengths generation
- limitations generation
- implementation steps
With RAG + LLM (Groq) integration
"""

import re
import json
from typing import Dict, List
from app.services.llm import call_groq

try:
    from app.rag.rag_engine import RAG_ENGINE
except:
    RAG_ENGINE = None


def _get_rag_context(query: str) -> str:
    if not query:
        return ""
    try:
        if RAG_ENGINE:
            ctx = RAG_ENGINE.get_context(query, k=6)
            return ctx or ""
    except:
        pass
    return ""

def _get_fallbacks(plan_data: Dict) -> Dict[str, Any]:
    """Return manual fallback data if AI fails"""
    s_name = plan_data.get("strategy", "")
    return {
        "strengths": ["Efisiensi operasional terukur", "Mitigasi risiko standar", "Alokasi sumber daya optimal"],
        "limitations": ["Bergantung pada akurasi data input", "Memerlukan monitoring manual", "Faktor eksternal tak terduga"],
        "implementation_steps": ["Lakukan briefing operasional", "Monitor KPI per shift", "Evaluasi harian"],
        "justification": f"Rencana {s_name} direkomendasikan berdasarkan perhitungan matematis efisiensi dan biaya terbaik saat ini."
    }

def generate_full_analysis(plan_data: Dict, domain: str, config: Dict) -> Dict[str, Any]:
    rag_context = _get_rag_context(plan_data.get("strategy"))
    
    prompt = f"""
    Bertindaklah sebagai AI Expert Senior di bidang {domain}.
    
    Konteks RAG: {rag_context}
    
    Tugas: Analisis rencana optimasi berikut dan berikan output lengkap.
    
    DATA RENCANA:
    - Strategi: {plan_data.get('strategy')} ({plan_data.get('description')})
    - Finansial: {json.dumps(plan_data.get('financial', {}))}
    
    Hasilkan output HANYA dalam format JSON valid dengan struktur berikut:
    {{
        "strengths": ["poin 1", "poin 2", "poin 3"],
        "limitations": ["poin 1", "poin 2", "poin 3"],
        "implementation_steps": ["langkah 1", "langkah 2", "langkah 3", "langkah 4"],
        "justification": "Paragraf narasi penjelasan profesional (maks 2 paragraf)..."
    }}
    
    PENTING:
    1. Bahasa Indonesia Profesional.
    2. Strengths: Fokus pada efisiensi dan mitigasi risiko.
    3. Limitations: Fokus pada margin error prediksi atau faktor manusia (JANGAN bilang kurang data cuaca).
    4. Steps: Langkah konkrit untuk foreman/supervisor.
    5. Justification: Mengapa rencana ini dipilih berdasarkan data trade-off risiko & biaya.
    
    JSON Output:
    """
    
    try:
        response = call_groq(prompt, config)
        
        cleaned = re.sub(r'```json\s*|\s*```', '', response).strip()
        
        parsed = json.loads(cleaned)
        
        fallbacks = _get_fallbacks(plan_data)
        return {
            "strengths": parsed.get("strengths", fallbacks["strengths"])[:4],
            "limitations": parsed.get("limitations", fallbacks["limitations"])[:4],
            "implementation_steps": parsed.get("implementation_steps", fallbacks["implementation_steps"])[:5],
            "justification": parsed.get("justification", fallbacks["justification"])
        }
        
    except Exception as e:
        print(f"AI Generation Error: {e}")
        return _get_fallbacks(plan_data)
