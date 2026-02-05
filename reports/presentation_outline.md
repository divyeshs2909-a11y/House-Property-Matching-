# Presentation Outline (Embeddings)

1. Objective: compute Match Score (0–100) for user-property pairs
2. Data: numeric prefs + narrative descriptions
3. Cleaning: numeric coercion + encoding fix (â€” etc.)
4. Text matching: semantic embeddings (SentenceTransformers or Gemini embedding)
5. Similarity: cosine similarity on normalized vectors
6. Numeric scores: budget fit + bed/bath need satisfaction
7. Final score: weighted blend + explain weights
8. Results: top-k per user, score distribution
9. Visuals: heatmap (users x properties), bar chart top matches for a user
10. Deployment: FastAPI + optional Streamlit UI
