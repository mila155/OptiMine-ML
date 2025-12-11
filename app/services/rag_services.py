from typing import List, Dict, Any
from app.rag.retriever import RAGRetriever  
import textwrap

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
    def __init__(self, retriever: RAGRetriever = None):
        self.retriever = retriever or RAGRetriever()

    def retrieve_context(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Return list of docs: each is {id, score, text, metadata}
        """
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

    def build_prompt(self, user_query: str, docs: List[Dict[str, Any]], history: List[Dict[str,str]] = None) -> str:
        ctx_parts = []
        for i, d in enumerate(docs):
            snippet = d["text"]
            if len(snippet) > 1500:
                snippet = snippet[:1500] + "..."  
            header = f"[SOURCE {i+1}] id={d.get('id')} score={d.get('score'):.3f} metadata={d.get('metadata')}"
            ctx_parts.append(header + "\n" + snippet)

        context_block = "\n\n".join(ctx_parts) if ctx_parts else "No relevant documents found."

        history_block = ""
        if history:
            hist_lines = []
            for h in history[-6:]:  
                msg = h.get("message", "")
                hist_lines.append(f"{role.upper()}: {msg}")
            history_block = "\n\nConversation history:\n" + "\n".join(hist_lines)

        prompt = f"""{SYSTEM_PROMPT}

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
        return prompt
