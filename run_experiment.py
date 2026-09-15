from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)

from lob.data import align_events, load_messages, load_orderbook, validate_book
from lob.evaluation import analyze_score_bins
from lob.features import (
    build_basic_features,
    compute_depth_imbalance,
    compute_rolling_features,
)
from lob.labels import make_targets
from lob.models import fit_baseline, fit_logistic
from lob.splits import make_temporal_splits, prepare_datasets

# Experiment configuration
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"

FILE_PREFIX = "AAPL_2012-06-21_34200000_57600000"
N_LEVELS = 10

HORIZON = 50
EPSILON = 0.0
WINDOWS = (10, 50, 100)
DEPTHS = (5, 10)

TRAIN_FRACTION = 0.6
VAL_FRACTION = 0.2

CLASSES = [-1, 0, 1]

FEATURE_COLUMNS = [
    "spread",
    "imbalance_1",
    *[f"imbalance_{depth}" for depth in DEPTHS],
    "weighted_mid_offset",
    *[
        f"{feature}_{window}"
        for window in WINDOWS
        for feature in ("log_return", "realized_vol")
    ],
]


def evaluate_model(model, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
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

    print("\nFull logistic regression — validation diagnostics:")
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


def main() -> None:
    # 1. Load and align the original event sequence.
    messages = load_messages(DATA_DIR / f"{FILE_PREFIX}_message_{N_LEVELS}.csv")
    book = load_orderbook(
        DATA_DIR / f"{FILE_PREFIX}_orderbook_{N_LEVELS}.csv",
        n_levels=N_LEVELS,
    )
    events = align_events(messages, book)

    print(f"Dataset: {FILE_PREFIX}")
    print(f"Events: {len(events):,} | Book levels: {N_LEVELS}")

    # 2. Check the book before computing features.
    checks = validate_book(book, n_levels=N_LEVELS)

    print("\nOrder book checks:")
    print(checks.sum().to_string())

    # This first experiment assumes complete, valid book snapshots.
    # Stop for inspection rather than silently removing events.
    if checks.any().any():
        raise ValueError("Order book flags detected; inspect before proceeding")

    # 3. Build features without dropping any events.
    features = build_basic_features(events)

    for depth in DEPTHS:
        features[f"imbalance_{depth}"] = compute_depth_imbalance(
            events,
            n_levels=depth,
        )

    historical = compute_rolling_features(
        features["mid_price"],
        windows=WINDOWS,
    )
    features = features.join(historical)

    # 4. Build targets from the original integer quote prices.
    label_mid = (events["bid_price_1"] + events["ask_price_1"]) / 20_000

    targets = make_targets(
        label_mid,
        horizon=HORIZON,
        epsilon=EPSILON,
    )

    # 5. Split the complete timeline, then remove unusable rows per split.
    splits = make_temporal_splits(
        n_events=len(events),
        horizon=HORIZON,
        train_fraction=TRAIN_FRACTION,
        val_fraction=VAL_FRACTION,
    )

    datasets = prepare_datasets(
        features=features,
        targets=targets,
        splits=splits,
        feature_columns=FEATURE_COLUMNS,
    )

    print(f"\nHorizon: {HORIZON} events | Epsilon: ${EPSILON:g}")
    print("Prepared datasets:")
    for name, (X, y) in datasets.items():
        print(f"  {name}: {len(y):,} observations, {X.shape[1]} features")

    X_train, y_train = datasets["train"]
    X_validation, y_validation = datasets["validation"]

    # 6. Fit models using training data only.
    baseline = fit_baseline(X_train, y_train)

    logistic_simple = fit_logistic(
        X_train[["imbalance_1"]],
        y_train,
    )

    logistic_full = fit_logistic(X_train, y_train)

    # 7. Compare models on validation. The test set remains untouched.
    experiments = [
        ("baseline", baseline, X_validation),
        ("logistic_simple", logistic_simple, X_validation[["imbalance_1"]]),
        ("logistic_full", logistic_full, X_validation),
    ]

    results = pd.DataFrame.from_dict(
        {
            name: evaluate_model(model, X_val, y_validation)
            for name, model, X_val in experiments
        },
        orient="index",
    )
    results.index.name = "model"

    print("\nValidation results:")
    print(results.round(6).to_string())

    print_classification_diagnostics(
        logistic_full,
        X_validation,
        y_validation,
    )
    score_summary = analyze_score_bins(
        logistic_full,
        X_validation,
        targets,
    )

    print("\nValidation — future changes by score bin:")
    print(score_summary.round(4).to_string())
    # Compute future quotes BEFORE selecting validation observations.
    quotes = pd.DataFrame(index=events.index)

    quotes["bid"] = events["bid_price_1"] / 10_000
    quotes["ask"] = events["ask_price_1"] / 10_000
    quotes["future_bid"] = quotes["bid"].shift(-HORIZON)
    quotes["future_ask"] = quotes["ask"].shift(-HORIZON)

    quotes = quotes.loc[X_validation.index].copy()

    probabilities = logistic_full.predict_proba(X_validation)
    classes = list(logistic_full.classes_)

    quotes["score"] = (
        probabilities[:, classes.index(1)] - probabilities[:, classes.index(-1)]
    )

    quotes["score_bin"] = pd.qcut(
        quotes["score"],
        q=10,
        labels=False,
        duplicates="drop",
    )

    quotes["spread"] = quotes["ask"] - quotes["bid"]

    # Hypothetical round-trip PnL per share, before explicit fees.
    quotes["long_pnl"] = quotes["future_bid"] - quotes["ask"]
    quotes["short_pnl"] = quotes["bid"] - quotes["future_ask"]

    execution_summary = quotes.groupby("score_bin").agg(
        count=("score", "size"),
        mean_spread=("spread", "mean"),
        mean_long_pnl=("long_pnl", "mean"),
        mean_short_pnl=("short_pnl", "mean"),
    )

    print("\nValidation — immediate execution diagnostic ($ per share):")
    print(execution_summary.round(4).to_string())


if __name__ == "__main__":
    main()
