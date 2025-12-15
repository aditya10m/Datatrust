import pandas as pd
from .metrics import (
    rounding_ratio,
    repetition_ratio,
    entropy_score,
    sudden_shift
)
from .ml import ml_anomaly_score

class TrustAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.results = {}

    def analyze(self):
        for col in self.df.columns:
            series = self.df[col]

            rep = repetition_ratio(series)
            round_bias = rounding_ratio(series) if series.dtype != "object" else 0
            ent = entropy_score(series)
            shift = sudden_shift(series) if series.dtype != "object" else 0
            ml_anomaly = ml_anomaly_score(series) if series.dtype != "object" else 0

            trust = 1 - (
                0.30 * rep +
                0.20 * round_bias +
                0.20 * (1 / (ent + 1)) +
                0.15 * shift +
                0.15 * ml_anomaly
            )

            trust = max(0, min(1, trust))

            self.results[col] = {
                "trust_score": round(trust, 3),
                "repetition": round(rep, 3),
                "rounding_bias": round(round_bias, 3),
                "entropy": round(ent, 3),
                "sudden_shift": round(shift, 3),
                "ml_anomaly": round(ml_anomaly, 3)
            }

        return self.results

    def trust_scores(self):
        if not self.results:
            self.analyze()
        return {k: v["trust_score"] for k, v in self.results.items()}
