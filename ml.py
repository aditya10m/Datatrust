import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

def ml_anomaly_score(series: pd.Series):
    numeric = series.dropna()

    if numeric.empty or numeric.nunique() < 5:
        return 0.0

    values = numeric.values.reshape(-1, 1)

    model = IsolationForest(
        contamination=0.1,
        random_state=42
    )
    model.fit(values)

    scores = model.decision_function(values)
    anomaly_ratio = (scores < 0).sum() / len(scores)

    return anomaly_ratio
