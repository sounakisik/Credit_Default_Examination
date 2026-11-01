import pandas as pd
from src.audit import data_quality_summary, duplicate_summary

def test_quality_summary_has_one_row_per_column():
    df = pd.DataFrame({"x": [1, 2, 3], "default": [0, 1, 0]})
    report = data_quality_summary(df, "default")
    assert len(report) == 2

def test_duplicate_summary():
    df = pd.DataFrame({"x": [1, 1, 2]})
    out = duplicate_summary(df)
    assert out["duplicate_rows"] == 1
