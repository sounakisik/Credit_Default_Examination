from __future__ import annotations
from sklearn.model_selection import train_test_split
import pandas as pd

def stratified_train_valid_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, stratify=y, random_state=random_state
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=random_state
    )
    return X_train, X_valid, X_test, y_train, y_valid, y_test
