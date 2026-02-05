import argparse
import json
import os
import numpy as np
import pandas as pd

from src.config import Config
from src.preprocess import clean_users, clean_properties
from src.scoring import price_score, need_score
from src.embeddings import load_or_compute_embeddings, cosine_matrix

def compute_match_scores(users: pd.DataFrame, props: pd.DataFrame, cfg: Config, text_backend: str) -> pd.DataFrame:
    user_ids = users["User ID"].astype(str).to_list()
    prop_ids = props["Property ID"].astype(str).to_list()

    # --- Embeddings (cached)
    cache_u = os.path.join(cfg.cache_dir, f"cache_users_{text_backend}.json")
    cache_p = os.path.join(cfg.cache_dir, f"cache_props_{text_backend}.json")

    u_emb = load_or_compute_embeddings(
        ids=user_ids,
        texts=users["Qualitative Description"].fillna("").astype(str).to_list(),
        backend=text_backend,  # type: ignore
        cache_path=cache_u,
        n_jobs=cfg.n_jobs,
    )
    p_emb = load_or_compute_embeddings(
        ids=prop_ids,
        texts=props["Qualitative Description"].fillna("").astype(str).to_list(),
        backend=text_backend,  # type: ignore
        cache_path=cache_p,
        n_jobs=cfg.n_jobs,
    )

    text_sim = cosine_matrix(u_emb, p_emb)  # users x props in [0,1]

    # --- Numeric scores (broadcast)
    budget = users["Budget"].to_numpy().reshape(-1, 1)
    price = props["Price"].to_numpy().reshape(1, -1)
    s_price = price_score(budget, price)

    u_bed = users["Bedrooms"].to_numpy().reshape(-1, 1)
    p_bed = props["Bedrooms"].to_numpy().reshape(1, -1)
    s_bed = need_score(u_bed, p_bed)

    u_bath = users["Bathrooms"].to_numpy().reshape(-1, 1)
    p_bath = props["Bathrooms"].to_numpy().reshape(1, -1)
    s_bath = need_score(u_bath, p_bath)

    # --- Weighted blend
    w = np.array([cfg.w_price, cfg.w_bed, cfg.w_bath, cfg.w_text], dtype=float)
    w = w / w.sum()

    score01 = w[0]*s_price + w[1]*s_bed + w[2]*s_bath + w[3]*text_sim
    score100 = np.round(score01 * 100.0, 2)

    # Long form output
    out = []
    for i, uid in enumerate(user_ids):
        for j, pid in enumerate(prop_ids):
            out.append({
                "User ID": uid,
                "Property ID": pid,
                "Match Score": float(score100[i, j]),
                "Price Score": float(np.round(s_price[i, j]*100, 2)),
                "Bedroom Score": float(np.round(s_bed[i, j]*100, 2)),
                "Bathroom Score": float(np.round(s_bath[i, j]*100, 2)),
                "Text Score": float(np.round(text_sim[i, j]*100, 2)),
            })

    df_out = pd.DataFrame(out)
    df_out["Match Score"] = pd.to_numeric(df_out["Match Score"], errors="raise")
    return df_out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--text_backend", type=str, default="sbert", choices=["sbert", "gemini"])
    args = ap.parse_args()

    cfg = Config()
    users = clean_users(pd.read_csv(cfg.users_path))
    props = clean_properties(pd.read_csv(cfg.properties_path))

    scores = compute_match_scores(users, props, cfg, text_backend=args.text_backend)
    scores = scores.sort_values(["User ID", "Match Score"], ascending=[True, False])
    scores_topk = scores.groupby("User ID").head(args.topk).reset_index(drop=True)

    scores_topk.to_csv(cfg.scores_out, index=False)

    summary = {
        "users": int(users.shape[0]),
        "properties": int(props.shape[0]),
        "rows_scored": int(scores.shape[0]),
        "rows_saved_topk": int(scores_topk.shape[0]),
        "topk": int(args.topk),
        "text_backend": args.text_backend,
        "weights_used_normalized": (np.array([cfg.w_price, cfg.w_bed, cfg.w_bath, cfg.w_text]) /
                                   np.sum([cfg.w_price, cfg.w_bed, cfg.w_bath, cfg.w_text])).round(4).tolist(),
        "cache_files": {
            "users": f"reports/cache_users_{args.text_backend}.json",
            "properties": f"reports/cache_props_{args.text_backend}.json",
        }
    }
    with open(cfg.summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved:", cfg.scores_out)
    print("Saved:", cfg.summary_out)

if __name__ == "__main__":
    main()
