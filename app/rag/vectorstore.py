# app/rag/vectorstore.py
import os
import numpy as np

try:
    from sentence_transformers import SentenceTransformer # type: ignore
    import faiss # type: ignore
except Exception as e:
    # If packages missing, raise a clear error when imported
    raise ImportError("Missing RAG deps: install sentence-transformers and faiss-cpu. Error: " + str(e))

class VectorStore:
    def __init__(self, docs_path: str):
        self.docs_path = docs_path
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.text_chunks = []
        self.embeddings = None

    def load_documents(self):
        docs = []
        for fname in sorted(os.listdir(self.docs_path)):
            if fname.endswith(".txt"):
                with open(os.path.join(self.docs_path, fname), "r", encoding="utf-8") as f:
                    text = f.read().strip()
                    # split on paragraphs/lines for finer granularity
                    for part in text.split("\n"):
                        text_chunk = part.strip()
                        if len(text_chunk) > 10:
                            docs.append({"source": fname, "text": text_chunk})
        return docs

    def build(self):
        docs = self.load_documents()
        self.text_chunks = [d["text"] for d in docs]
        if not self.text_chunks:
            raise RuntimeError("No documents found to build vector store in: " + self.docs_path)

        self.embeddings = np.array(self.model.encode(self.text_chunks)).astype("float32")
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)
        # store source mapping
        self.sources = [d["source"] for d in docs]

    def search(self, query: str, top_k: int = 5):
        if self.index is None:
            raise RuntimeError("Vector index not built. Call build() first.")
        q_emb = np.array(self.model.encode([query])).astype("float32")
        D, I = self.index.search(q_emb, top_k)
        results = []
        for idx in I[0]:
            if idx >= 0 and idx < len(self.text_chunks):
                results.append({
                    "text": self.text_chunks[idx],
                    "source": self.sources[idx]
                })
        return results
