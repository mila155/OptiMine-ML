import os
import google.generativeai as genai

from dotenv import load_dotenv
load_dotenv()

# Load API key from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

def call_groq(prompt: str, config: dict) -> str:
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        response = model.generate_content(
            prompt,
            safety_settings=safety_settings,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3, 
                max_output_tokens=2000
            )
        )
        
        return response.text
        
    except Exception as e:
        print(f"GEMINI ERROR: {e}")
        return '{"justification": "Maaf, analisis AI sedang sibuk. Data ditampilkan berdasarkan perhitungan manual."}'
