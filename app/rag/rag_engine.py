# app/rag/rag_engine.py
from app.rag.retriever import RAGRetriever
from app.rag.vectorstore import VectorStore


class RAGEngineSafe:
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
        except Exception as e:
            print("RAG init error:", e)
            self.vs = None
            self.retriever = None

    def get_context(self, query: str, k: int = 5) -> str:
        if not query:
            return ""
        try:
            if self.retriever:
                return self.retriever.get_context(query, top_k=k)
        except Exception:
            pass
        return ""