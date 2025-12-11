import json
from typing import Dict, Any
from app.services.llm import call_groq

DEFAULT_LLM_CONFIG = {
    "model": "llama-3.3-70b-versatile",
    "temperature": 0.2,
    "max_tokens": 800
}

class LLMService:

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or DEFAULT_LLM_CONFIG

    def ask(self, prompt: str, override_config: Dict[str, Any] = None) -> str:
        cfg = self.config.copy()
        if override_config:
            cfg.update(override_config)
        return call_groq(prompt, cfg)

