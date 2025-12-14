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

def _parse_ai_list_response(response_text: str) -> List[str]:
    try:
        cleaned = re.sub(r'```json\s*|\s*```', '', response_text).strip()
        cleaned = re.sub(r'```python\s*|\s*```', '', cleaned).strip()
        cleaned = re.sub(r'```\s*|\s*```', '', cleaned).strip()
        
        match = re.search(r'\[.*\]', cleaned, re.DOTALL)
        if match:
            json_str = match.group(0)
            parsed = json.loads(json_str)
            clean_list = []
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, str):
                        clean_list.append(item)
                    elif isinstance(item, dict):
                        vals = [str(v) for v in item.values() if isinstance(v, str)]
                        if vals:
                            preferred = item.get('deskripsi') or item.get('tindakan') or max(vals, key=len)
                            clean_list.append(str(preferred))
                        else:
                            clean_list.append(str(item))
                            
            return clean_list[:5] if clean_list else []
        
        lines = [line.strip('- *').strip() for line in response_text.split('\n') if line.strip()]
        if len(lines) >= 3:
            return lines[:5] 
            
        return []
    except Exception as e:
        print(f"Error parsing AI response: {e}")
        return []

def generate_strengths_ai(plan_data: Dict, domain: str, config: Dict) -> List[str]:
    rag_context = _get_rag_context(plan_data.get("strategy"))

    prompt = f"""
       Gunakan konteks berikut: {rag_context}
       Analisis rencana: "({plan_data.get('strategy')})" ({plan_data.get('description')})
       Impact: {json.dumps(plan_data.get('financial', {}))}
       
       Kamu adalah AI yang memberikan penilaian professional mengenai rencana optimasi {domain}.
       
       Berdasarkan data berikut:
       - Strategy: {plan_data.get("strategy")}
       - Financial: {plan_data.get("financial")}
       - Jumlah schedule item: {len(plan_data.get("schedule", []))}
       
       Tuliskan 3 kelebihan (strengths) yang spesifik, realistis, teknis, dan relevan dalam bahasa Indonesia Formal.
       Konteks Sistem: Sistem ini SUDAH menggunakan data real-time (cuaca, alat, dll).
       Gunakan format list JSON: ["...", "...", "..."].
       """

    try:
        resp = call_groq(prompt, config)
        return eval(resp) if resp.startswith("[") else ["Strength 1", "Strength 2", "Strength 3"]
    except:
        return ["Strength 1", "Strength 2", "Strength 3"]


def generate_limitations_ai(plan_data: Dict, domain: str, config: Dict) -> List[str]:
    rag_context = _get_rag_context(plan_data.get("strategy"))

    prompt = f"""
Gunakan konteks berikut:
{rag_context}
Analisis risiko strategi: {plan_data.get('strategy')}

Kamu adalah AI yang memberikan analisa risiko dan keterbatasan rencana optimasi {domain}.

PENTING: 
    - Sistem ini SUDAH memperhitungkan cuaca dan kondisi jalan. 
    - JANGAN tulis "tidak memperhitungkan faktor eksternal/cuaca".
    
Berdasarkan data berikut:
- Strategy: {plan_data.get("strategy")}
- Financial: {plan_data.get("financial")}
- Schedule items: {len(plan_data.get("schedule", []))}

Tuliskan 3 keterbatasan (limitations) yang objektif, teknis, dan masuk akal dalam bahasa Indonesia Formal.
Format list JSON saja.
"""

    try:
        resp = call_groq(prompt, config)
        return eval(resp) if resp.startswith("[") else ["Limitation 1", "Limitation 2", "Limitation 3"]
    except:
        return ["Limitation 1", "Limitation 2", "Limitation 3"]


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
        result = _parse_ai_list_response(resp)
        if result: return result       
    except:
        pass
        
    return _get_fallback_steps(plan_data.get("strategy", ""))

def _get_fallback_steps(plan_name: str) -> List[str]:
    plan_lower = plan_name.lower()
    
    if "conservative" in plan_lower or "saver" in plan_lower or "weather" in plan_lower:
        return [
            "Kurangi target produksi harian pada shift malam",
            "Prioritaskan maintenance preventif alat utama",
            "Tingkatkan monitoring cuaca per jam",
            "Siapkan area dumping cadangan yang lebih dekat",
            "Lakukan briefing keselamatan khusus cuaca basah"
        ]
    elif "aggressive" in plan_lower or "boost" in plan_lower or "max" in plan_lower:
        return [
            "Optimalkan pergantian shift (hot seat change)",
            "Aktifkan unit cadangan (standby units)",
            "Tingkatkan kecepatan hauling di segmen jalan lurus",
            "Prioritaskan loading point dengan cycle time terpendek",
            "Monitor fuel burn rate secara ketat"
        ]
    else: 
        return [
            "Jalankan operasi sesuai baseline schedule",
            "Monitor KPI per 4 jam dan sesuaikan jika ada deviasi",
            "Optimalkan antrean di ROM/Jetty",
            "Pastikan ketersediaan operator > 95%",
            "Lakukan evaluasi harian pasca-operasi"
        ]
    
def generate_strengths_ai(plan_data: Dict, domain: str, config: Dict) -> List[str]:
    print(f"Generating strengths for {plan_data.get('strategy')}...")
    
    rag_context = _get_rag_context(plan_data.get("strategy"))
    
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
        print(f"Raw AI response for strengths: {resp}")
        
        import re
        import json
        
        cleaned = re.sub(r'```json\s*|\s*```', '', resp).strip()
        
        match = re.search(r'\[.*\]', cleaned, re.DOTALL)
        if match:
            json_str = match.group(0)
            strengths = json.loads(json_str)
            
            if isinstance(strengths, list) and len(strengths) > 0:
                strengths = [str(item) for item in strengths[:3]]
                print(f"Parsed strengths: {strengths}")
                return strengths
        
        return _generate_fallback_strengths(plan_data, domain)
        
    except Exception as e:
        print(f"Error generating strengths: {e}")
        return _generate_fallback_strengths(plan_data, domain)

def _generate_fallback_strengths(plan_data: Dict, domain: str) -> List[str]:
    fin = plan_data.get("financial", {})
    savings = fin.get('cost_savings_usd', 0)
    risk = fin.get('avg_risk_score', 0.5)
    
    strengths = []
    
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
        if savings < 0:  
            strengths.append("Investasi untuk peningkatan produktivitas jangka panjang")
    
    else:
        strengths.append("Rencana yang terstruktur dan terukur")
        strengths.append("Mempertimbangkan berbagai faktor operasional")
        strengths.append("Dapat diimplementasikan dengan kontrol risiko")
    
    return strengths[:3]  
