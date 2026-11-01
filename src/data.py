from __future__ import annotations
import pandas as pd
from ucimlrepo import fetch_ucirepo

def load_credit_default_data() -> tuple[pd.DataFrame, pd.Series]:
    dataset = fetch_ucirepo(id=350)
    X = dataset.data.features.copy()
    y = dataset.data.targets.iloc[:, 0].astype(int)
    y.name = "default"
    return X, y

def build_modeling_frame() -> pd.DataFrame:
    X, y = load_credit_default_data()
    df = X.copy()
    df[y.name] = y.values
    return df
