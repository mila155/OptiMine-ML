"""
AI helper functions for optimization plan enrichment
Handles:
- strengths generation
- limitations generation
- implementation steps
With RAG + LLM (Groq) integration
"""

from typing import Dict, List
from app.services.llm import call_groq

# Try RAG engine
try:
    from app.rag.rag_engine import RAG_ENGINE
except:
    RAG_ENGINE = None


def _get_rag_context(query: str) -> str:
    """Fetch context from RAG engine (safe)."""
    if not query:
        return ""
    try:
        if RAG_ENGINE:
            ctx = RAG_ENGINE.get_context(query, k=6)
            return ctx or ""
    except:
        pass
    return ""


# ============================================================
# 1. Strengths
# ============================================================
def generate_strengths_ai(plan_data: Dict, domain: str, config: Dict) -> List[str]:
    rag_context = _get_rag_context(plan_data.get("strategy"))

    prompt = f"""
Gunakan konteks berikut:
{rag_context}

Kamu adalah AI yang memberikan penilaian professional mengenai rencana optimasi {domain}.

Berdasarkan data berikut:
- Strategy: {plan_data.get("strategy")}
- Financial: {plan_data.get("financial")}
- Jumlah schedule item: {len(plan_data.get("schedule", []))}

Tuliskan 3 kelebihan (strengths) yang spesifik, realistis, teknis, dan relevan.
Gunakan format list JSON: ["...", "...", "..."].
"""

    try:
        resp = call_groq(prompt, config)
        return eval(resp) if resp.startswith("[") else ["Strength 1", "Strength 2", "Strength 3"]
    except:
        return ["Strength 1", "Strength 2", "Strength 3"]


# ============================================================
# 2. Limitations
# ============================================================
def generate_limitations_ai(plan_data: Dict, domain: str, config: Dict) -> List[str]:
    rag_context = _get_rag_context(plan_data.get("strategy"))

    prompt = f"""
Gunakan konteks berikut:
{rag_context}

Kamu adalah AI yang memberikan analisa risiko dan keterbatasan rencana optimasi {domain}.

Berdasarkan data berikut:
- Strategy: {plan_data.get("strategy")}
- Financial: {plan_data.get("financial")}
- Schedule items: {len(plan_data.get("schedule", []))}

Tuliskan 3 keterbatasan (limitations) yang objektif, teknis, dan masuk akal.
Format list JSON saja.
"""

    try:
        resp = call_groq(prompt, config)
        return eval(resp) if resp.startswith("[") else ["Limitation 1", "Limitation 2", "Limitation 3"]
    except:
        return ["Limitation 1", "Limitation 2", "Limitation 3"]


# ============================================================
# 3. Implementation Steps
# ============================================================
def generate_steps_ai(plan_data: Dict, domain: str, config: Dict) -> List[str]:
    rag_context = _get_rag_context(plan_data.get("strategy"))

    prompt = f"""
Gunakan konteks berikut:
{rag_context}

Kamu adalah AI yang membuat langkah implementasi profesional untuk rencana optimasi {domain}.

Data:
- Strategy: {plan_data.get("strategy")}
- Financial: {plan_data.get("financial")}

Tuliskan 3-5 langkah implementasi yang actionable, spesifik, dan operasional.
Format JSON list.
"""

    try:
        resp = call_groq(prompt, config)
        return eval(resp) if resp.startswith("[") else ["Step 1", "Step 2", "Step 3"]
    except:
        return ["Step 1", "Step 2", "Step 3"]