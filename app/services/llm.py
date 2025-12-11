import os
from groq import Groq

from dotenv import load_dotenv
load_dotenv()

# Load API key from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Create client
groq_client = Groq(api_key=GROQ_API_KEY)

def call_groq(prompt: str, config: dict) -> str:
    try:
        response = groq_client.chat.completions.create(
            model=config.get("model", "llama-3.3-70b-versatile"),
            messages=[{"role": "user", "content": prompt}],
            temperature=config.get("temperature", 0.2),
            max_tokens=config.get("max_tokens", 1024)
        )
        return response.choices[0].message["content"]

    except Exception as e:
        # Tambahkan logging yang jelas untuk Railway log
        print("⚠️ [LLM ERROR] Could not call Groq API:", e)
        raise e  # <-- jangan fallback diam-diam

