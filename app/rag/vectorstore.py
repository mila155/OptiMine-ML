import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
from transformers import AutoTokenizer, AutoModel
import torch

class VectorStore:
    def __init__(self, docs_path: str, model_name="intfloat/e5-small"):
        self.docs_path = docs_path
        self.model_name = model_name

        # Load HF model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        self.text_chunks = []
        self.embeddings = None
        self.index = None
        self.sources = []

    # -----------------------------
    # LOAD DOCS
    # -----------------------------
    def load_documents(self):
        docs = []
        for fname in sorted(os.listdir(self.docs_path)):
            if fname.endswith(".txt"):
                with open(os.path.join(self.docs_path, fname), "r", encoding="utf-8") as f:
                    text = f.read().strip()
                    for part in text.split("\n"):
                        chunk = part.strip()
                        if len(chunk) > 10:
                            docs.append({"source": fname, "text": chunk})
        return docs

    # -----------------------------
    # ENCODER
    # -----------------------------
    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Mean pooling
        emb = outputs.last_hidden_state.mean(dim=1)

        return emb.cpu().numpy()

    # -----------------------------
    # BUILD VECTOR STORE
    # -----------------------------
    def build(self):
        docs = self.load_documents()
        self.text_chunks = [d["text"] for d in docs]
        self.sources = [d["source"] for d in docs]

        if not self.text_chunks:
            raise RuntimeError("No documents found in: " + self.docs_path)

        # Compute embeddings
        self.embeddings = self.encode(self.text_chunks)

        # Build sklearn NN index
        self.index = NearestNeighbors(n_neighbors=5, metric="cosine")
        self.index.fit(self.embeddings)

        print("VectorStore (light) built successfully.")

    # -----------------------------
    # SEARCH
    # -----------------------------
    def search(self, query: str, top_k: int = 5):
        if self.index is None:
            raise RuntimeError("Index not built. Call build() first.")

        q_emb = self.encode(query)

        distances, indices = self.index.kneighbors(q_emb, n_neighbors=top_k)

        results = []
        for idx in indices[0]:
            results.append({
                "text": self.text_chunks[idx],
                "source": self.sources[idx]
            })

        return results