
"""Adapter around the existing Resolvr pipeline (src/pipeline.py).
 
Heavy objects (embedding model, embedding matrix) are loaded once at startup.
`process` is blocking (Groq calls + model inference), so the API runs it in a
thread pool and never on the event loop.
"""
import pickle
import sys
from dataclasses import dataclass
from typing import List, Tuple
 
TOP_K = 3
 
 
@dataclass
class Result:
    intent: str
    auto_handle: bool
    reason: str
    reply: str
    grounding: List[Tuple[int, float]]  # (grounding_pairs.id, cosine similarity), best first
 
 
class Engine:
    def __init__(self, src_dir: str, data_dir: str):
        import numpy as np  # heavy deps are imported here so tests can import Result without them
 
        sys.path.insert(0, src_dir)
        import pipeline  # noqa: E402  (needs GROQ_API_KEY in the environment)
        from sentence_transformers import SentenceTransformer
 
        self._pipeline = pipeline
        with open(f"{data_dir}/grounding_pairs.pkl", "rb") as f:
            self.pairs = pickle.load(f)
        with open(f"{data_dir}/grounding_embeddings.pkl", "rb") as f:
            emb = np.asarray(pickle.load(f), dtype=np.float32)
        self._emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)  # normalise once
        self._embed_model = SentenceTransformer("all-MiniLM-L6-v2")
 
    def _retrieve(self, text: str) -> List[Tuple[int, float]]:
        import numpy as np
 
        q = self._embed_model.encode([self._pipeline.clean_text(text)])[0]
        q = q / (np.linalg.norm(q) + 1e-9)
        sims = self._emb @ q
        top = np.argsort(sims)[-TOP_K:][::-1]
        return [(int(i), float(sims[i])) for i in top]
 
    def process(self, text: str) -> Result:
        p = self._pipeline
        intent = p.classify_intent(text)
        grounding = self._retrieve(text)
        reply = p.draft_reply(text, intent, [self.pairs[i] for i, _ in grounding])
        auto_handle, reason = p.decide_escalation(intent, text)
        return Result(intent, bool(auto_handle), reason, reply, grounding)
 