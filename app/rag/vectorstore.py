import os

from sklearn.neighbors import NearestNeighbors
from app.rag.embedder import SimpleEmbedder


class VectorStore:
    def __init__(self, docs_path: str):
        self.docs_path = docs_path
        self.embedder = SimpleEmbedder()
        self.index = None
        self.text_chunks = []
        self.sources = []

    def load_documents(self):
        docs = []
        if not os.path.exists(self.docs_path):
            return [{"source": "fallback", "text": "Dokumen tidak ditemukan"}]

        for fname in sorted(os.listdir(self.docs_path)):
            if fname.endswith(".txt"):
                filepath = os.path.join(self.docs_path, fname)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        text = f.read().strip()
                        # split per baris, abaikan terlalu pendek
                        for part in text.split("\n"):
                            line = part.strip()
                            if len(line) > 3:  # lebih rendah threshold
                                docs.append({"source": fname, "text": line})
                except:
                    continue
        if not docs:
            # fallback jika folder kosong
            docs.append({"source": "fallback", "text": "Dokumen kosong, gunakan fallback konteks"})
        return docs

    def build(self):
        docs = self.load_documents()
        self.text_chunks = [d["text"] for d in docs]
        self.sources = [d["source"] for d in docs]

        # Fit embedder
        self.embedder.fit(self.text_chunks)
        embeddings = self.embedder.encode(self.text_chunks)

        # Build Nearest Neighbor index (safe n_neighbors)
        n_neighbors = min(5, len(embeddings))
        self.index = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
        self.index.fit(embeddings)
        self.embeddings = embeddings

    def search(self, query: str, top_k: int = 5):
        if not self.index or not self.text_chunks:
            # fallback
            return [{"text": "Tidak ada konteks tersedia", "source": "fallback"}]

        query_vec = self.embedder.encode([query])
        k = min(top_k, len(self.text_chunks))
        distances, indices = self.index.kneighbors(query_vec, n_neighbors=k)

        results = []
        for idx in indices[0]:
            results.append({
                "text": self.text_chunks[idx],
                "source": self.sources[idx]
            })
        if not results:
            results = [{"text": "Tidak ada konteks relevan tersedia", "source": "fallback"}]
        return results


# import os
# import numpy as np
# from groq import Groq

# class VectorStore:
#     def __init__(self, docs_path: str, chunk_size: int = 350):
#         self.docs_path = docs_path
#         self.chunk_size = chunk_size
#         self.chunks = []
#         self.embeddings = []
#         self.client = Groq()

#     def _chunk_text(self, text: str):
#         words = text.split()
#         for i in range(0, len(words), self.chunk_size):
#             yield " ".join(words[i:i+self.chunk_size])

#     def embed_text(self, text: str):
#         resp = self.client.embeddings.create(
#             model="text-embedding-3-large",
#             input=text
#         )
#         return np.array(resp.data[0].embedding, dtype=np.float32)

#     def cosine_sim(a, b):
#         return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
#     def load_documents(self):
#         docs = []
#         for fname in sorted(os.listdir(self.docs_path)):
#             if fname.endswith(".txt"):
#                 with open(os.path.join(self.docs_path, fname), "r", encoding="utf-8") as f:
#                     full = f.read().strip()
#                 for chunk in self._chunk_text(full):
#                     docs.append({"source": fname, "text": chunk})
#         return docs

#     def build(self):
#         docs = self.load_documents()
#         for d in docs:
#             emb = self.embed_text(d["text"])
#             self.chunks.append(d)
#             self.embeddings.append(emb)

#         self.embeddings = np.vstack(self.embeddings)

#     def search(self, query: str, top_k: int = 5):
#         query_emb = self.embed_text(query)

#         sims = np.array([
#             cosine_sim(q_emb, e) for e in self.embeddings
#         ])
        
#         top_idx = sims.argsort()[::-1][:top_k]

#         results = []
#         for i in top_idx:
#             results.append({
#                 "text": self.text_chunks[i]["text"],
#                 "source": self.text_chunks[i]["source"],
#                 "score": float(sims[i])
#             })
#         return results