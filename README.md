# Property Matching — Semantic Embeddings (Best for narrative descriptions)

This version upgrades matching to **semantic embeddings** (better than TF‑IDF for long, story-like text).

## Data
- `data/users.csv` columns:
  `User ID,Budget,Bedrooms,Bathrooms,Qualitative Description`
- `data/properties.csv` columns:
  `Property ID,Price,Bedrooms,Bathrooms,Living Area (sq ft),Qualitative Description`

## Install
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Batch scoring (top-k per user)
```bash
python -m src.score --topk 10 --text_backend sbert
```

Text backends:
- `sbert` (default): local SentenceTransformers (no API cost)
- `gemini`: Gemini Embedding API (needs env var `GEMINI_API_KEY`)

Example:
```bash
export GEMINI_API_KEY="YOUR_KEY"
python -m src.score --topk 10 --text_backend gemini
```

Outputs:
- `reports/match_scores.csv`
- `reports/summary.json`
- cached embeddings under `reports/cache_*` (speeds re-runs)

## API (optional)
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8001 --workers 4
```

## UI (optional)
```bash
streamlit run ui/app.py
```

## Why embeddings are better here
Narrative descriptions often have low keyword overlap (TF‑IDF fails).
Embeddings match by **meaning** (e.g., “cozy retreat” ~ “warm and inviting home”). 
- SentenceTransformers: strong baseline, fully local.
- Gemini embedding model exists on Gemini API tiers. 
