from __future__ import annotations
import pandas as pd

def data_quality_summary(df: pd.DataFrame, target: str) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col]
        rows.append({
            "feature": col,
            "dtype": str(s.dtype),
            "n_obs": len(s),
            "n_missing": int(s.isna().sum()),
            "missing_rate": float(s.isna().mean()),
            "n_unique": int(s.nunique(dropna=True)),
            "min": s.min() if pd.api.types.is_numeric_dtype(s) else None,
            "max": s.max() if pd.api.types.is_numeric_dtype(s) else None,
            "mean": s.mean() if pd.api.types.is_numeric_dtype(s) else None,
            "is_target": col == target,
        })
    return pd.DataFrame(rows)

def duplicate_summary(df: pd.DataFrame) -> dict:
    return {
        "n_rows": len(df),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_rate": float(df.duplicated().mean()),
    }

def target_summary(df: pd.DataFrame, target: str) -> pd.DataFrame:
    counts = df[target].value_counts(dropna=False).sort_index()
    out = pd.DataFrame({"count": counts, "rate": counts / counts.sum()})
    out.index.name = target
    return out

def suspicious_cardinality(df: pd.DataFrame, target: str) -> pd.DataFrame:
    rows = []
    n = len(df)
    for col in df.columns:
        if col == target:
            continue
        nunique = df[col].nunique(dropna=False)
        rows.append({
            "feature": col,
            "n_unique": nunique,
            "unique_fraction": nunique / max(n, 1),
            "constant_flag": nunique <= 1,
            "id_like_flag": nunique / max(n, 1) > 0.95,
        })
    return pd.DataFrame(rows).sort_values(
        ["id_like_flag", "unique_fraction"], ascending=[False, False]
    )
