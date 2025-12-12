from app.rag.vectorstore import VectorStore


class RAGRetriever:
    def __init__(self, vectorstore: VectorStore):
        self.store = vectorstore

    def get_context(self, query: str, top_k: int = 5):
        results = self.store.search(query, top_k)
        if not results:
            return "Tidak ada konteks relevan tersedia."
        return "\n".join([r["text"] for r in results])
