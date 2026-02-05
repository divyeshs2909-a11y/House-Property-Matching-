import os
import json
import numpy as np
from typing import List, Literal, Tuple
from joblib import Parallel, delayed

from src.text_clean import fix_mojibake

TextBackend = Literal["sbert", "gemini"]

def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / denom

def embed_texts_sbert(texts: List[str], model_name: str = "all-MiniLM-L6-v2", batch_size: int = 64) -> np.ndarray:
    # Local embeddings (no API cost). all-MiniLM-L6-v2 is fast + good baseline. 
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_name)
    texts = [fix_mojibake(t) for t in texts]
    emb = m.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=False)
    emb = np.asarray(emb, dtype=np.float32)
    return _l2_normalize(emb)

def embed_texts_gemini(texts: List[str], model: str = "gemini-embedding-001", batch_size: int = 64, n_jobs: int = 8) -> np.ndarray:
    # Gemini embeddings via Google GenAI SDK. Requires env GEMINI_API_KEY.
    # Docs: https://ai.google.dev/gemini-api/docs/embeddings
    from google import genai
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is required for text_backend=gemini")

    client = genai.Client(api_key=api_key)

    texts = [fix_mojibake(t) for t in texts]

    def one(t: str):
        res = client.models.embed_content(model=model, contents=t)
        # SDK returns embedding as list[float]
        return np.asarray(res.embeddings[0].values, dtype=np.float32)

    # Parallel requests (multicore) — keep n_jobs modest to avoid quota errors
    vecs = Parallel(n_jobs=n_jobs, backend="threading")(delayed(one)(t) for t in texts)
    mat = np.vstack(vecs)
    return _l2_normalize(mat)

def load_or_compute_embeddings(
    ids: List[str],
    texts: List[str],
    backend: TextBackend,
    cache_path: str,
    n_jobs: int = -1,
) -> np.ndarray:
    # Cache file stores: {"ids": [...], "emb": [[...], ...]}
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if obj.get("ids") == list(ids):
            emb = np.asarray(obj["emb"], dtype=np.float32)
            # ensure normalized
            denom = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
            return emb / denom

    if backend == "sbert":
        emb = embed_texts_sbert(texts)
    elif backend == "gemini":
        # choose a safe default for concurrency
        emb = embed_texts_gemini(texts, n_jobs=8 if n_jobs == -1 else min(16, max(1, n_jobs)))
    else:
        raise ValueError(f"Unknown backend: {backend}")

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"ids": list(ids), "emb": emb.tolist()}, f)

    return emb

def cosine_matrix(user_emb: np.ndarray, prop_emb: np.ndarray) -> np.ndarray:
    # Both normalized -> cosine = dot product
    return np.clip(user_emb @ prop_emb.T, 0.0, 1.0)
