import os
import numpy as np

try:
    from sentence_transformers import SentenceTransformer 
    import faiss 
except Exception as e:
    raise ImportError("Missing RAG deps: install sentence-transformers and faiss-cpu. Error: " + str(e))

class VectorStore:
    def __init__(self, docs_path: str):
        self.docs_path = docs_path
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.text_chunks = []
        self.embeddings = []
        self.client = Groq()

    def embed_text(self, text: str):
        resp = self.client.embeddings.create(
            model="llama-3.3-70b-versatile",
            input=text
        )
        return np.array(resp.data[0].embedding)

    def load_documents(self):
        docs = []
        for fname in sorted(os.listdir(self.docs_path)):
            if fname.endswith(".txt"):
                with open(os.path.join(self.docs_path, fname), "r", encoding="utf-8") as f:
                    docs.append({
                        "source": fname,
                        "text": f.read().strip()
                    })
        return docs

    def build(self):
        docs = self.load_documents()
        for d in docs:
            emb = self.embed_text(d["text"])
            self.text_chunks.append(d)
            self.embeddings.append(emb)

        self.embeddings = np.array(self.embeddings)

    def search(self, query: str, top_k: int = 5):
        query_emb = self.embed_text(query)

        sims = (self.embeddings @ query_emb) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_emb)
        )

        top_idx = sims.argsort()[::-1][:top_k]

        results = []
        for i in top_idx:
            results.append({
                "text": self.text_chunks[i]["text"],
                "source": self.text_chunks[i]["source"],
                "score": float(sims[i])
            })
        return results
