import os
from groq import Groq

from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

groq_client = Groq(api_key=GROQ_API_KEY)

def call_groq(prompt: str, config: dict) -> str:
    try:
        response = groq_client.chat.completions.create(
            model=config.get("model", "llama-3.3-70b-versatile"),
            messages=[{"role": "user", "content": prompt}],
            temperature=config.get("temperature", 0.2),
            max_tokens=config.get("max_tokens", 1024)
        )
        return response.choices[0].message.content
    except Exception as e:
        print("⚠️ LLM ERROR:", e)
        return f"Penjelasan otomatis gagal: {e}"
