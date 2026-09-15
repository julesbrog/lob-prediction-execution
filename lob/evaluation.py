import pandas as pd


def analyze_score_bins(
    model,
    X: pd.DataFrame,
    targets: pd.DataFrame,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Summarize future price changes by prediction-score quantile."""

    probabilities = model.predict_proba(X)
    classes = list(model.classes_)

    p_down = probabilities[:, classes.index(-1)]
    p_up = probabilities[:, classes.index(1)]

    analysis = pd.DataFrame(
        {
            "score": p_up - p_down,
            "future_change": targets.loc[X.index, "future_change"],
            "target": targets.loc[X.index, "target"],
        },
        index=X.index,
    )

    if analysis.isna().any().any():
        raise ValueError("score analysis contains missing values")

    analysis["score_bin"] = pd.qcut(
        analysis["score"],
        q=n_bins,
        labels=False,
        duplicates="drop",
    )

    analysis["is_up"] = analysis["target"].eq(1).astype(float)
    analysis["is_down"] = analysis["target"].eq(-1).astype(float)
    analysis["is_flat"] = analysis["target"].eq(0).astype(float)

    summary = analysis.groupby("score_bin").agg(
        count=("score", "size"),
        mean_score=("score", "mean"),
        mean_change_dollars=("future_change", "mean"),
        median_change_dollars=("future_change", "median"),
        up_fraction=("is_up", "mean"),
        down_fraction=("is_down", "mean"),
        flat_fraction=("is_flat", "mean"),
    )

    return summary
