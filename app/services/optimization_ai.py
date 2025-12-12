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
    
def generate_strengths_ai(plan_data: Dict, domain: str, config: Dict) -> List[str]:
    print(f"🔄 Generating strengths for {plan_data.get('strategy')}...")
    
    rag_context = _get_rag_context(plan_data.get("strategy"))
    
    # Gunakan prompt yang lebih spesifik
    prompt = f"""
Gunakan konteks berikut:
{rag_context}

Kamu adalah AI yang memberikan penilaian profesional mengenai rencana optimasi {domain}.

Berdasarkan data berikut:
- Strategy: {plan_data.get("strategy")}
- Financial: {plan_data.get("financial")}
- Jumlah schedule item: {len(plan_data.get("schedule", []))}

Tuliskan TIGA kelebihan (strengths) yang spesifik, realistis, teknis, dan relevan untuk industri pertambangan.
**HANYA keluarkan format JSON array tanpa penjelasan lain.**
Contoh: ["Keuntungan pertama", "Keuntungan kedua", "Keuntungan ketiga"]

JSON array:
"""
    
    try:
        resp = call_groq(prompt, config)
        print(f"📥 Raw AI response for strengths: {resp}")
        
        # Bersihkan response
        import re
        import json
        
        # Hapus markdown code blocks jika ada
        cleaned = re.sub(r'```json\s*|\s*```', '', resp).strip()
        
        # Cari array JSON dalam response
        match = re.search(r'\[.*\]', cleaned, re.DOTALL)
        if match:
            json_str = match.group(0)
            strengths = json.loads(json_str)
            
            if isinstance(strengths, list) and len(strengths) > 0:
                # Pastikan semua item adalah string
                strengths = [str(item) for item in strengths[:3]]
                print(f"✅ Parsed strengths: {strengths}")
                return strengths
        
        # Jika parsing gagal, gunakan fallback berdasarkan data
        return _generate_fallback_strengths(plan_data, domain)
        
    except Exception as e:
        print(f"❌ Error generating strengths: {e}")
        return _generate_fallback_strengths(plan_data, domain)

def _generate_fallback_strengths(plan_data: Dict, domain: str) -> List[str]:
    """Generate fallback strengths berdasarkan data plan"""
    fin = plan_data.get("financial", {})
    savings = fin.get('cost_savings_usd', 0)
    risk = fin.get('avg_risk_score', 0.5)
    
    strengths = []
    
    # Berdasarkan tipe plan
    plan_name = plan_data.get("strategy", "").lower()
    
    if "conservative" in plan_name:
        strengths.append("Fokus pada mitigasi risiko dan stabilitas operasional")
        if savings > 0:
            strengths.append(f"Penghematan biaya sebesar ${savings:,.0f}")
        strengths.append("Mengurangi ketergantungan pada kondisi cuaca ekstrem")
        strengths.append("Jadwal operasi yang lebih dapat diprediksi")
    
    elif "balanced" in plan_name:
        strengths.append("Keseimbangan optimal antara target produksi dan manajemen risiko")
        strengths.append("Fleksibilitas dalam menyesuaikan operasi harian")
        strengths.append("Mempertahankan efisiensi operasional rata-rata")
        strengths.append("Kemampuan adaptasi terhadap perubahan kondisi")
    
    elif "aggressive" in plan_name:
        strengths.append("Maksimalisasi output produksi untuk permintaan tinggi")
        strengths.append("Pemanfaatan optimal kapasitas alat berat")
        strengths.append("Potensi peningkatan revenue yang signifikan")
        if savings < 0:  # Biaya lebih tinggi
            strengths.append("Investasi untuk peningkatan produktivitas jangka panjang")
    
    else:
        strengths.append("Rencana yang terstruktur dan terukur")
        strengths.append("Mempertimbangkan berbagai faktor operasional")
        strengths.append("Dapat diimplementasikan dengan kontrol risiko")
    
    return strengths[:3]  # Maksimal 3 items