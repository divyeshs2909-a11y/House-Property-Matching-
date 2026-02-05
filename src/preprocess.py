import pandas as pd
from src.text_clean import fix_mojibake

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df

import re
import pandas as pd
import numpy as np

_MULT = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}

def to_numeric(series: pd.Series) -> pd.Series:
    def parse_one(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return np.nan
        s = str(x).strip().lower()

        # remove currency symbols/words and commas/spaces
        s = s.replace(",", "")
        s = s.replace("$", "").replace("usd", "").replace("inr", "").replace("₹", "")
        s = s.replace("rs.", "").replace("rs", "")

        # match formats like: 500k, 1.2m, 750000
        m = re.match(r"^(-?\d+(\.\d+)?)([kmb])?$", s)
        if not m:
            # fallback: keep only digits/dot/minus then parse
            s2 = re.sub(r"[^0-9.\-]", "", s)
            return float(s2) if s2 not in ("", "-", ".", "-.") else np.nan

        num = float(m.group(1))
        suf = m.group(3)
        return num * _MULT[suf] if suf else num

    return series.apply(parse_one)


def clean_users(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df).copy()
    df["Budget"] = to_numeric(df["Budget"])
    df["Bedrooms"] = to_numeric(df["Bedrooms"])
    df["Bathrooms"] = to_numeric(df["Bathrooms"])
    df["Qualitative Description"] = df.get("Qualitative Description", "").fillna("").map(fix_mojibake)
    for c in ["Budget", "Bedrooms", "Bathrooms"]:
        df[c] = df[c].fillna(df[c].median())
    return df

def clean_properties(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df).copy()
    df["Price"] = to_numeric(df["Price"])
    df["Bedrooms"] = to_numeric(df["Bedrooms"])
    df["Bathrooms"] = to_numeric(df["Bathrooms"])
    if "Living Area (sq ft)" in df.columns:
        df["Living Area (sq ft)"] = to_numeric(df["Living Area (sq ft)"])
    df["Qualitative Description"] = df.get("Qualitative Description", "").fillna("").map(fix_mojibake)
    for c in ["Price", "Bedrooms", "Bathrooms"]:
        df[c] = df[c].fillna(df[c].median())
    if "Living Area (sq ft)" in df.columns:
        df["Living Area (sq ft)"] = df["Living Area (sq ft)"].fillna(df["Living Area (sq ft)"].median())
    return df
