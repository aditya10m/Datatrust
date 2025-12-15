def explain_column(col, metrics):
    reasons = []

    if metrics["repetition"] > 0.5:
        reasons.append("High value repetition detected")

    if metrics["rounding_bias"] > 0.6:
        reasons.append("Suspicious rounding patterns")

    if metrics["entropy"] < 1.0:
        reasons.append("Low entropy (possible fabricated data)")

    if metrics["sudden_shift"] > 0.4:
        reasons.append("Sudden distribution shift")

    if not reasons:
        return f"{col} looks trustworthy."

    return f"{col} flagged: " + "; ".join(reasons)
