from typing import List, Dict, Any
import textwrap

from app.rag.vectorstore import VectorStore
from app.rag.retriever import RAGRetriever  

SYSTEM_PROMPT = textwrap.dedent("""\
You are OptiMine AI Assistant — an expert in mining production, hauling logistics,
ROM/ROM-to-jetty operations and shipping planning.

Rules:
- Answer only using the provided CONTEXT. If context does not contain the answer,
  respond: "Saya tidak menemukan informasi tersebut dalam dokumen."
- Avoid hallucination. Do not invent numbers or facts.
- Provide concise, actionable output. Use short bullets when giving recommendations.
- If user asks for predictions or to run models, point them to /mining/predict or /shipping/predict.
""")

class RAGService:
    def __init__(self, docs_path: str = "app/rag/documents"):
        # ✅ build vectorstore di sini
        self.vs = VectorStore(docs_path)
        self.vs.build()

        # ✅ retriever selalu dapat vectorstore
        self.retriever = RAGRetriever(self.vs)

    def retrieve_context(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        docs = self.retriever.retrieve(query, top_k=top_k)

        results = []
        for d in docs:
            text = getattr(d, "page_content", None) or getattr(d, "text", None) or str(d)
            meta = getattr(d, "metadata", {}) if hasattr(d, "metadata") else {}
            doc_id = meta.get("id") or getattr(d, "id", None) or meta.get("source", None)
            score = float(getattr(d, "score", 0.0) or 0.0)

            results.append({
                "id": doc_id,
                "score": score,
                "text": text,
                "metadata": meta
            })
        return results

    def build_prompt(
        self,
        user_query: str,
        docs: List[Dict[str, Any]],
        history: List[Dict[str, str]] = None
    ) -> str:
        ctx_parts = []
        for i, d in enumerate(docs):
            snippet = d["text"][:1500] + ("..." if len(d["text"]) > 1500 else "")
            header = (
                f"[SOURCE {i+1}] id={d.get('id')} "
                f"score={d.get('score'):.3f} metadata={d.get('metadata')}"
            )
            ctx_parts.append(header + "\n" + snippet)

        context_block = "\n\n".join(ctx_parts) if ctx_parts else "No relevant documents found."

        history_block = ""
        if history:
            hist_lines = [
                f"{h.get('role', 'User').upper()}: {h.get('message', '')}"
                for h in history[-6:]
            ]
            history_block = "\n\nConversation history:\n" + "\n".join(hist_lines)

        return f"""{SYSTEM_PROMPT}

CONTEXT:
{context_block}

{history_block}

USER QUESTION:
{user_query}

INSTRUCTIONS:
- Use ONLY the CONTEXT above to answer. If the answer is not present, say exactly:
  "Saya tidak menemukan informasi tersebut dalam dokumen."
- Be concise (3-8 sentences) and provide actionable recommendations if relevant.
- If you cite a fact from a document, include the source id in square brackets, e.g. [SOURCE 1].
- Output plain text (no JSON).
"""
