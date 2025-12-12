# -------------------------------
# rag_engine.py
# -------------------------------
from app.rag.vectorstore import VectorStore
from app.rag.retriever import RAGRetriever


class RAGEngineSafe:
    """
    RAG Engine Safe:
    - Mengambil konteks dari dokumen secara aman
    - Jika error / dokumen kosong / query invalid, tetap return string aman
    """
    def __init__(self, docs_path="app/rag/documents"):
        self.docs_path = docs_path
        self.vs = None
        self.retriever = None
        self._init_engine()

    def _init_engine(self):
        try:
            self.vs = VectorStore(self.docs_path)
            self.vs.build()
            self.retriever = RAGRetriever(self.vs)
        except Exception:
            self.vs = None
            self.retriever = None
def get_context(self, query: str, k: int = 5) -> str:
    if not query:
        return "Tidak ada konteks tersedia"
    try:
        if self.retriever:
            ctx = self.retriever.get_context(query, top_k=k)
            return ctx or "Tidak ada konteks tersedia"
    except Exception:
        pass
    return "Tidak ada konteks tersedia"  # fallback aman