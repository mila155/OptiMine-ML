import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
from .embedder import SimpleEmbedder


class VectorStore:
    def __init__(self, docs_path: str):
        self.docs_path = docs_path
        self.embedder = SimpleEmbedder()
        self.index = None
        self.text_chunks = []

    def load_documents(self):
        docs = []
        for fname in sorted(os.listdir(self.docs_path)):
            if fname.endswith(".txt"):
                filepath = os.path.join(self.docs_path, fname)
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                    for part in text.split("\n"):
                        line = part.strip()
                        if len(line) > 10:
                            docs.append({"source": fname, "text": line})
        return docs

    def build(self):
        docs = self.load_documents()
        self.text_chunks = [d["text"] for d in docs]
        self.sources = [d["source"] for d in docs]

        # Fit TF-IDF embedders
        self.embedder.fit(self.text_chunks)
        embeddings = self.embedder.encode(self.text_chunks)

        # Build Nearest Neighbor index
        self.index = NearestNeighbors(
            n_neighbors=5,
            metric="cosine"
        )
        self.index.fit(embeddings)
        self.embeddings = embeddings

    def search(self, query: str, top_k: int = 5):
        query_vec = self.embedder.encode([query])
        distances, indices = self.index.kneighbors(query_vec, n_neighbors=top_k)

        results = []
        for idx in indices[0]:
            results.append({
                "text": self.text_chunks[idx],
                "source": self.sources[idx]
            })

        return results