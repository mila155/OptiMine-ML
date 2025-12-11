import json
from app.services.llm import call_groq

DEFAULT_LLM_CONFIG = {
    "model": "llama-3.3-70b-versatile",
    "temperature": 0.2,
    "max_tokens": 800
}

class LLMService:

    def summarize_mining(self, summary: dict) -> str:       
        prompt = f"""
Anda adalah AI Mining Operations Advisor.
Analisis data summary berikut dan buatkan ringkasan operasional untuk Mining Planner:

DATA:
{json.dumps(summary, indent=2, default=str)}

TUGAS ANDA:
1. Berikan penilaian umum terhadap kinerja produksi pada periode tersebut.
2. Analisis faktor yang memengaruhi produksi: cuaca, efisiensi, dan risiko.
3. Jelaskan tren gap produksi dan apa dampaknya bagi operasi harian.
4. Identifikasi hari berisiko tinggi dan penyebabnya.
5. Berikan rekomendasi praktis:
   - Pengaturan ulang prioritas operasi
   - Antisipasi keterlambatan akibat cuaca
   - Penyesuaian target produksi
6. Gunakan angka konkret dalam ringkasan.

GAYA BAHASA:
- Natural dan profesional seperti mining consultant berpengalaman.
- Maksimal 200 kata.

OUTPUT:
Hanya teks ringkasan final.
"""

        return call_groq(prompt, DEFAULT_LLM_CONFIG)

    def summarize_shipping(self, summary: dict) -> str:
        prompt = f"""
Anda adalah AI Shipping Operations Advisor.
Analisis summary berikut dan buatkan ringkasan operasional untuk SHIPPING PLANNER:

DATA:
{json.dumps(summary, indent=2, default=str)}

TUGAS ANDA:
1. Berikan penilaian hauling dan vessel loading.
2. Identifikasi risiko demurrage dan hari berisiko tinggi.
3. Analisis bottleneck: jetty congestion, loading efficiency, cuaca.
4. Berikan rekomendasi praktis:
   - Rescheduling vessel
   - Adjust hauling rate
   - Risk mitigation
5. Maksimal 200 kata.

OUTPUT:
Hanya teks ringkasan final.
"""
        return call_groq(prompt, DEFAULT_LLM_CONFIG)
