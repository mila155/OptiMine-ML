class RAGRetriever:
    def __init__(self, vector_store):
        self.vs = vector_store

    def get_context(self, query: str, k: int = 5):
        try:
            hits = self.vs.search(query, top_k=k)
            return "\n".join([f"[{h['source']}] {h['text']}" for h in hits])
        except:
            return ""
