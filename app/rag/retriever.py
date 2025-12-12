class RAGRetriever:
    def __init__(self, vectorstore):
        self.store = vectorstore

    def get_context(self, query: str, top_k: int = 5):
        results = self.store.search(query, top_k)
        return "\n".join([r["text"] for r in results])