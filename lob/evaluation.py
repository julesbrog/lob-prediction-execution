import pandas as pd
import numpy as np
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


def decompose_trade_pnl(
    events: pd.DataFrame,
    trades: pd.DataFrame,
) -> pd.DataFrame:
    """Decompose top-of-book round-trip PnL into mid movement and costs."""

    result = trades.copy()

    new_columns = [
        "mid_pnl",
        "spread_cost",
        "reconstructed_net_pnl",
    ]

    if result.empty:
        for column in new_columns:
            result[column] = pd.Series(index=result.index, dtype=float)
        return result

    if not events.columns.is_unique or not trades.columns.is_unique:
        raise ValueError("column names must be unique")

    required_event_cols = ["bid_price_1", "ask_price_1"]
    required_trade_cols = [
        "entry_index",
        "exit_index",
        "side",
        "quantity",
        "fees",
        "net_pnl",
    ]

    for column in required_event_cols:
        if column not in events.columns:
            raise ValueError(f"missing event column: {column}")

    for column in required_trade_cols:
        if column not in trades.columns:
            raise ValueError(f"missing trade column: {column}")

    for column in ("entry_index", "exit_index"):
        indices = result[column]

        if (
            not pd.api.types.is_integer_dtype(indices.dtype)
            or pd.api.types.is_bool_dtype(indices.dtype)
            or indices.isna().any()
        ):
            raise ValueError(f"{column} must contain integer positions")

        if ((indices < 0) | (indices >= len(events))).any():
            raise ValueError(f"{column} is outside the available history")

    entry_indices = result["entry_index"].to_numpy(dtype=int)
    exit_indices = result["exit_index"].to_numpy(dtype=int)

    if np.any(exit_indices <= entry_indices):
        raise ValueError("exits must occur after entries")

    # NumPy arrays preserve transaction order without pandas index alignment.
    entry_quotes = events.iloc[entry_indices]
    exit_quotes = events.iloc[exit_indices]

    entry_bid = entry_quotes["bid_price_1"].to_numpy(dtype=float)
    entry_ask = entry_quotes["ask_price_1"].to_numpy(dtype=float)
    exit_bid = exit_quotes["bid_price_1"].to_numpy(dtype=float)
    exit_ask = exit_quotes["ask_price_1"].to_numpy(dtype=float)

    for bid, ask in ((entry_bid, entry_ask), (exit_bid, exit_ask)):
        if not np.isfinite(bid).all() or not np.isfinite(ask).all():
            raise ValueError("quote prices must be finite")

        if (
            np.any(bid <= 0)
            or np.any(ask <= 0)
            or np.any(ask == 9999999999)
            or np.any(ask < bid)
        ):
            raise ValueError("invalid best quotes")

    side = result["side"].to_numpy(dtype=float)
    quantity = result["quantity"].to_numpy(dtype=float)
    fees = result["fees"].to_numpy(dtype=float)
    net_pnl = result["net_pnl"].to_numpy(dtype=float)

    if not np.isin(side, [-1, 1]).all():
        raise ValueError("side must be -1 or 1")

    if (
        not np.isfinite(quantity).all()
        or np.any(quantity <= 0)
        or np.any(quantity != np.floor(quantity))
    ):
        raise ValueError("quantities must be positive integers")

    if not np.isfinite(fees).all() or np.any(fees < 0):
        raise ValueError("fees must be finite and non-negative")

    if not np.isfinite(net_pnl).all():
        raise ValueError("net_pnl must be finite")

    # Convert raw LOBSTER quote prices into dollars.
    entry_mid = (entry_bid + entry_ask) / 20_000
    exit_mid = (exit_bid + exit_ask) / 20_000

    entry_spread = (entry_ask - entry_bid) / 10_000
    exit_spread = (exit_ask - exit_bid) / 10_000

    result["mid_pnl"] = side * quantity * (exit_mid - entry_mid)
    result["spread_cost"] = quantity * (entry_spread + exit_spread) / 2

    result["reconstructed_net_pnl"] = result["mid_pnl"] - result["spread_cost"] - fees

    np.testing.assert_allclose(
        result["reconstructed_net_pnl"].to_numpy(),
        net_pnl,
        rtol=1e-9,
        atol=1e-9,
        err_msg="PnL decomposition does not match recorded net PnL",
    )

    return result
