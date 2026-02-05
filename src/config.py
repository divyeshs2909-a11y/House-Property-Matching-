from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    users_path: str = "data/User.csv"
    properties_path: str = "data/Properties.csv"

    scores_out: str = "reports/match_scores.csv"
    summary_out: str = "reports/summary.json"

    # Weights (normalized automatically)
    w_price: float = 0.40
    w_bed: float = 0.15
    w_bath: float = 0.10
    w_text: float = 0.35

    # Embedding cache paths
    cache_dir: str = "reports"
    n_jobs: int = -1
