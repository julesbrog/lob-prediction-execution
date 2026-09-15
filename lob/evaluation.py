import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)

CLASSES = [-1, 0, 1]


def evaluate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict[str, float]:
    """Evaluate a fitted classifier on one dataset."""

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    return {
        "log_loss": log_loss(
            y,
            probabilities,
            labels=model.classes_,
        ),
        "accuracy": accuracy_score(y, predictions),
        "macro_f1": f1_score(
            y,
            predictions,
            labels=CLASSES,
            average="macro",
            zero_division=0,
        ),
    }


def print_classification_diagnostics(
    model,
    X: pd.DataFrame,
    y: pd.Series,
) -> None:
    """Print per-class metrics and a row-normalized confusion matrix."""

    predictions = model.predict(X)

    print("\nClassification diagnostics:")
    print(
        classification_report(
            y,
            predictions,
            labels=CLASSES,
            target_names=["down", "flat", "up"],
            digits=4,
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        y,
        predictions,
        labels=CLASSES,
        normalize="true",
    )

    print("Confusion matrix — proportions within each actual class:")
    print(
        pd.DataFrame(
            matrix,
            index=["actual_down", "actual_flat", "actual_up"],
            columns=["pred_down", "pred_flat", "pred_up"],
        ).round(3)
    )


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

    return analysis.groupby("score_bin").agg(
        count=("score", "size"),
        mean_score=("score", "mean"),
        mean_change_dollars=("future_change", "mean"),
        median_change_dollars=("future_change", "median"),
        up_fraction=("is_up", "mean"),
        down_fraction=("is_down", "mean"),
        flat_fraction=("is_flat", "mean"),
    )


def analyze_aggressive_execution(
    model,
    X: pd.DataFrame,
    events: pd.DataFrame,
    horizon: int,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Analyze hypothetical immediate round trips before fees and latency.

    Assumes one complete trading session in original event order.
    Overlapping hypothetical trades are evaluated independently.
    """

    # Compute future quotes on the complete event sequence.
    quotes = pd.DataFrame(index=events.index)

    quotes["bid"] = events["bid_price_1"] / 10_000
    quotes["ask"] = events["ask_price_1"] / 10_000
    quotes["future_bid"] = quotes["bid"].shift(-horizon)
    quotes["future_ask"] = quotes["ask"].shift(-horizon)

    # Select evaluated observations only after shifting.
    quotes = quotes.loc[X.index].copy()

    probabilities = model.predict_proba(X)
    classes = list(model.classes_)

    quotes["score"] = (
        probabilities[:, classes.index(1)] - probabilities[:, classes.index(-1)]
    )

    quotes["score_bin"] = pd.qcut(
        quotes["score"],
        q=n_bins,
        labels=False,
        duplicates="drop",
    )

    quotes["spread"] = quotes["ask"] - quotes["bid"]

    # The spread is already included in these execution prices.
    quotes["long_pnl"] = quotes["future_bid"] - quotes["ask"]
    quotes["short_pnl"] = quotes["bid"] - quotes["future_ask"]

    return quotes.groupby("score_bin").agg(
        count=("score", "size"),
        mean_spread=("spread", "mean"),
        mean_long_pnl=("long_pnl", "mean"),
        mean_short_pnl=("short_pnl", "mean"),
    )
