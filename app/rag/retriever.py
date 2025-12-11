# app/rag/retriever.py
class RAGRetriever:
    def __init__(self, vector_store):
        self.vs = vector_store

    def get_context(self, query: str, k: int = 5):
        try:
            hits = self.vs.search(query, top_k=k)
            # join with source markers
            context_pieces = []
            for h in hits:
                context_pieces.append(f"[{h['source']}] {h['text']}")
            return "\n".join(context_pieces)
        except Exception as e:
            return ""  