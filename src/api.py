from typing import Optional, Literal
import os
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from src.config import Config
from src.scoring import price_score, need_score
from src.embeddings import embed_texts_sbert, embed_texts_gemini, cosine_matrix

cfg = Config()
app = FastAPI(title="Property Match Score API (Embeddings)", version="1.0")

TextBackend = Literal["sbert", "gemini"]

class UserPref(BaseModel):
    User_ID: str
    Budget: float
    Bedrooms: float
    Bathrooms: float
    Qualitative_Description: Optional[str] = ""

class PropertyChar(BaseModel):
    Property_ID: str
    Price: float
    Bedrooms: float
    Bathrooms: float
    Qualitative_Description: Optional[str] = ""

class MatchRequest(BaseModel):
    user: UserPref
    property: PropertyChar
    text_backend: TextBackend = "sbert"

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/match")
def match(req: MatchRequest):
    u = req.user
    p = req.property

    if req.text_backend == "sbert":
        u_emb = embed_texts_sbert([u.Qualitative_Description or ""])
        p_emb = embed_texts_sbert([p.Qualitative_Description or ""])
    else:
        u_emb = embed_texts_gemini([u.Qualitative_Description or ""])
        p_emb = embed_texts_gemini([p.Qualitative_Description or ""])

    text_sim = float(cosine_matrix(u_emb, p_emb)[0, 0])

    s_price = float(price_score(np.array([[u.Budget]]), np.array([[p.Price]]))[0, 0])
    s_bed = float(need_score(np.array([[u.Bedrooms]]), np.array([[p.Bedrooms]]))[0, 0])
    s_bath = float(need_score(np.array([[u.Bathrooms]]), np.array([[p.Bathrooms]]))[0, 0])

    w = np.array([cfg.w_price, cfg.w_bed, cfg.w_bath, cfg.w_text], dtype=float)
    w = w / w.sum()
    score01 = w[0]*s_price + w[1]*s_bed + w[2]*s_bath + w[3]*text_sim
    score100 = float(np.round(score01 * 100.0, 2))

    return {
        "user_id": u.User_ID,
        "property_id": p.Property_ID,
        "match_score": score100,
        "components": {
            "price": round(s_price*100, 2),
            "bedrooms": round(s_bed*100, 2),
            "bathrooms": round(s_bath*100, 2),
            "text": round(text_sim*100, 2),
        },
        "text_backend": req.text_backend,
    }
