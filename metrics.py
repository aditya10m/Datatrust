import numpy as np
import pandas as pd
from scipy.stats import entropy

def rounding_ratio(series: pd.Series):
    numeric = series.dropna()
    if numeric.empty:
        return 0.0
    rounded = numeric.astype(str).str.endswith(".0").sum()
    return rounded / len(numeric)

def repetition_ratio(series: pd.Series):
    counts = series.value_counts(normalize=True)
    return counts.iloc[0] if not counts.empty else 0.0

def entropy_score(series: pd.Series):
    counts = series.value_counts()
    probs = counts / counts.sum()
    return entropy(probs)

def sudden_shift(series: pd.Series):
    numeric = series.dropna()
    if len(numeric) < 10:
        return 0.0
    first = numeric.iloc[: len(numeric)//2].mean()
    second = numeric.iloc[len(numeric)//2 :].mean()
    return abs(second - first) / (abs(first) + 1e-6)
