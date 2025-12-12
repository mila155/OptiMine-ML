from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

class Embedder:
    def __init__(self, model_name="intfloat/e5-small"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)

        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.cpu().numpy()
