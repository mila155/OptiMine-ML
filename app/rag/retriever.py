from app.rag.vectorstore import VectorStore

class RAGRetriever:
    def __init__(self, vectorstore: VectorStore):
        self.store = vectorstore   # konsisten dengan branch main

    def get_context(self, query: str, top_k: int = 5):
        try:
            results = self.store.search(query, top_k)
            if not results:
                return "Tidak ada konteks relevan tersedia."

            return "\n".join([f"[{r['source']}] {r['text']}" for r in results])
        
        except Exception:
            return "Tidak ada konteks tersedia."
