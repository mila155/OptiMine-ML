import os
import google.generativeai as genai

from dotenv import load_dotenv
load_dotenv()

def call_groq(prompt: str, config: dict) -> str:
    try:
        if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
            del os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
            
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("CRITICAL: GOOGLE_API_KEY tidak ditemukan!")
            return '{"justification": "Error: API Key Google tidak ditemukan. Cek Variables di Railway."}'
            
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        
        generation_config = genai.types.GenerationConfig(
            temperature=config.get("temperature", 0.3) if config else 0.3,
            max_output_tokens=2000
        )

        response = model.generate_content(
            prompt,
            safety_settings=safety_settings,
            generation_config=generation_config
        )
        
        return response.text
        
    except Exception as e:
        print(f"GEMINI ERROR: {e}")
        return '{"justification": "Maaf, analisis AI sedang sibuk. Data ditampilkan berdasarkan perhitungan manual."}'
