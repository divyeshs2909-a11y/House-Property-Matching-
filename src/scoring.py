import numpy as np

def clamp01(x):
    return np.clip(x, 0.0, 1.0)

def price_score(budget: np.ndarray, price: np.ndarray) -> np.ndarray:
    budget = np.maximum(budget, 1.0)
    ratio = price / budget
    under = 1.0 - np.abs(ratio - 0.95) / 0.95
    under = clamp01(under)
    over = np.exp(-3.0 * (ratio - 1.0))
    over = clamp01(over)
    return np.where(ratio <= 1.0, under, over)

def need_score(need: np.ndarray, have: np.ndarray, tolerance: float = 0.0) -> np.ndarray:
    shortfall = np.maximum(0.0, (need - have) - tolerance)
    score = 1.0 - 0.5 * shortfall
    return clamp01(score)
