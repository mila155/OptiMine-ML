# app/rag/rag_engine.py
from app.rag.vectorstore import VectorStore
from app.rag.retriever import RAGRetriever

class RAGEngine:
    def __init__(self, docs_path="app/rag/documents"):
        self.vs = VectorStore(docs_path)
        self.vs.build()
        self.retriever = RAGRetriever(self.vs)

    def get_context(self, query: str, k: int = 5):
        return self.retriever.get_context(query, k=k)