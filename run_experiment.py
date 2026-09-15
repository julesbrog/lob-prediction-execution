from pathlib import Path

import numpy as np
import pandas as pd

from lob.data import (
    align_events,
    load_messages,
    load_orderbook,
    validate_book,
)
from lob.evaluation import (
    analyze_aggressive_execution,
    analyze_score_bins,
    evaluate_model,
    print_classification_diagnostics,
)
from lob.features import (
    build_basic_features,
    compute_depth_imbalance,
    compute_event_ofi,
    compute_rolling_features,
    compute_rolling_ofi,
)
from lob.labels import make_targets
from lob.models import fit_baseline, fit_boosting, fit_logistic
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

OFI_FEATURE_COLUMNS = FEATURE_COLUMNS + [f"ofi_{window}" for window in WINDOWS]


def main() -> None:
    # 1. Load and align events.
    messages = load_messages(DATA_DIR / f"{FILE_PREFIX}_message_{N_LEVELS}.csv")
    book = load_orderbook(
        DATA_DIR / f"{FILE_PREFIX}_orderbook_{N_LEVELS}.csv",
        n_levels=N_LEVELS,
    )
    events = align_events(messages, book)

    print(f"Dataset: {FILE_PREFIX}")
    print(f"Events: {len(events):,} | Book levels: {N_LEVELS}")

    # 2. Validate the book.
    checks = validate_book(book, n_levels=N_LEVELS)

    print("\nOrder book checks:")
    print(checks.sum().to_string())

    if checks.any().any():
        raise ValueError("Order book flags detected; inspect before proceeding")

    # 3. Build the original features.
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

    # 4. Build OFI features on the complete event sequence.
    event_ofi = compute_event_ofi(events)
    rolling_ofi = compute_rolling_ofi(
        event_ofi,
        windows=WINDOWS,
    )

    if not rolling_ofi.index.equals(features.index):
        raise ValueError("OFI and other features must have the same index")

    features = features.join(rolling_ofi)

    print("\nRolling OFI — missing values:")
    print(rolling_ofi.isna().sum().to_string())

    # 5. Build targets from raw integer quote prices.
    label_mid = (events["bid_price_1"] + events["ask_price_1"]) / 20_000

    targets = make_targets(
        label_mid,
        horizon=HORIZON,
        epsilon=EPSILON,
    )

    # 6. Create common temporal boundaries.
    splits = make_temporal_splits(
        n_events=len(events),
        horizon=HORIZON,
        train_fraction=TRAIN_FRACTION,
        val_fraction=VAL_FRACTION,
    )

    # Prepare both feature sets using the same boundaries.
    datasets = prepare_datasets(
        features=features,
        targets=targets,
        splits=splits,
        feature_columns=FEATURE_COLUMNS,
    )

    datasets_ofi = prepare_datasets(
        features=features,
        targets=targets,
        splits=splits,
        feature_columns=OFI_FEATURE_COLUMNS,
    )

    # Verify that comparisons use identical observations and labels.
    for name in splits:
        X_reference, y_reference = datasets[name]
        X_with_ofi, y_with_ofi = datasets_ofi[name]

        if not X_reference.index.equals(X_with_ofi.index):
            raise ValueError(
                f"{name}: reference and OFI datasets have different events"
            )

        if not y_reference.equals(y_with_ofi):
            raise ValueError(
                f"{name}: reference and OFI datasets have different labels"
            )

        np.testing.assert_allclose(
            X_reference.to_numpy(),
            X_with_ofi[FEATURE_COLUMNS].to_numpy(),
        )

    print(f"\nHorizon: {HORIZON} events | Epsilon: ${EPSILON:g}")
    print("Prepared datasets:")

    for name in splits:
        X_reference, y_reference = datasets[name]
        X_with_ofi, _ = datasets_ofi[name]

        print(
            f"  {name}: {len(y_reference):,} observations | "
            f"reference: {X_reference.shape[1]} features | "
            f"with OFI: {X_with_ofi.shape[1]} features"
        )

    X_train, y_train = datasets["train"]
    X_validation, y_validation = datasets["validation"]

    X_train_ofi, y_train_ofi = datasets_ofi["train"]
    X_validation_ofi, y_validation_ofi = datasets_ofi["validation"]

    # 7. Fit all models using training data only.
    baseline = fit_baseline(X_train, y_train)

    logistic_simple = fit_logistic(
        X_train[["imbalance_1"]],
        y_train,
    )

    logistic_full = fit_logistic(X_train, y_train)

    logistic_ofi = fit_logistic(X_train_ofi, y_train_ofi)

    boosting_ofi = fit_boosting(X_train_ofi, y_train_ofi)

    # 8. Compare all four models on validation.
    experiments = [
        ("baseline", baseline, X_validation, y_validation),
        (
            "logistic_simple",
            logistic_simple,
            X_validation[["imbalance_1"]],
            y_validation,
        ),
        (
            "logistic_full",
            logistic_full,
            X_validation,
            y_validation,
        ),
        (
            "logistic_ofi",
            logistic_ofi,
            X_validation_ofi,
            y_validation_ofi,
        ),
        (
            "boosting_ofi",
            boosting_ofi,
            X_validation_ofi,
            y_validation_ofi,
        ),
    ]

    results = pd.DataFrame.from_dict(
        {
            name: evaluate_model(model, X_val, y_val)
            for name, model, X_val, y_val in experiments
        },
        orient="index",
    )
    results.index.name = "model"

    print("\nValidation results:")
    print(results.round(6).to_string())

    # 9. Keep detailed diagnostics on the original 11-feature model.
    print("\nGradientBoosting model (14 features) — validation:")
    print_classification_diagnostics(
        boosting_ofi,
        X_validation_ofi,
        y_validation_ofi,
    )

    score_summary = analyze_score_bins(
        model=boosting_ofi,
        X=X_validation_ofi,
        targets=targets,
    )

    print("\nGradientBoosting model — future changes by score bin:")
    print(score_summary.round(4).to_string())

    execution_summary = analyze_aggressive_execution(
        model=boosting_ofi,
        X=X_validation_ofi,
        events=events,
        horizon=HORIZON,
    )

    print("\nGradientBoosting model — immediate execution diagnostic ($ per share):")
    print(execution_summary.round(4).to_string())

    # The test sets are prepared but are not used for model evaluation.


if __name__ == "__main__":
    main()
