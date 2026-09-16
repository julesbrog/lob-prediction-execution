from pathlib import Path

import numpy as np
import pandas as pd

from lob.backtest import (
    run_backtest,
    run_backtest_with_latency,
)
from lob.data import (
    align_events,
    compute_horizon_duration,
    load_messages,
    load_orderbook,
    validate_book,
)
from lob.evaluation import (
    analyze_aggressive_execution,
    analyze_score_bins,
    decompose_trade_pnl,
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

SIGNAL_THRESHOLD = 0.3
TRADE_QUANTITY = 1
FEE_PER_SHARE = 0.0

LATENCY_SCENARIOS = (0.0, 0.001, 0.010)


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

    horizon_seconds = compute_horizon_duration(
        events,
        horizon=HORIZON,
    )

    for name in ("train", "validation"):
        X, _ = datasets[name]
        durations = horizon_seconds.loc[X.index]

        print(f"\n{name} — duration of {HORIZON} events, in seconds:")
        print(
            durations.describe(percentiles=[0.01, 0.10, 0.50, 0.90, 0.99]).to_string()
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

    # 8. Compare all five models on validation.
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

    # 9. Detailed validation diagnostics for the boosting model.
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

    # Build prediction scores while preserving original event indices.
    probabilities = boosting_ofi.predict_proba(X_validation_ofi)
    classes = list(boosting_ofi.classes_)

    validation_scores = pd.Series(
        probabilities[:, classes.index(1)] - probabilities[:, classes.index(-1)],
        index=X_validation_ofi.index,
        name="score",
    )

    # Simulate non-overlapping round trips on validation.
    trades = run_backtest(
        events=events,
        scores=validation_scores,
        horizon=HORIZON,
        threshold=SIGNAL_THRESHOLD,
        quantity=TRADE_QUANTITY,
        fee_per_share=FEE_PER_SHARE,
    )

    print("\nBoosting + OFI — validation backtest")
    print("Zero latency; fees per share per transaction:", FEE_PER_SHARE)
    print("Number of trades:", len(trades))

    if trades.empty:
        print("No trades executed.")
    else:
        print(trades.head().to_string(index=False))

        print("\nTotal gross PnL ($):", trades["gross_pnl"].sum())
        print("Total fees ($):", trades["fees"].sum())
        print("Total net PnL ($):", trades["net_pnl"].sum())
        print("Mean net PnL per trade ($):", trades["net_pnl"].mean())
        print("Winning trade fraction:", trades["net_pnl"].gt(0).mean())

        print("\nNet PnL by position side (-1: short, +1: long):")
        print(
            trades.groupby("side")["net_pnl"].agg(["count", "sum", "mean"]).to_string()
        )
    decomposed_trades = decompose_trade_pnl(events, trades)

    print("\nValidation — PnL decomposition ($):")
    print(
        decomposed_trades[["mid_pnl", "spread_cost", "fees", "reconstructed_net_pnl"]]
        .sum()
        .round(4)
        .to_string()
    )
    # Restrict execution data to events strictly before the test block.
    val_end = splits["test"].start
    execution_events = events.iloc[:val_end].copy()

    period_end = float(execution_events["time"].iloc[-1])
    max_latency = max(LATENCY_SCENARIOS)

    # Shared boundaries for every latency scenario.
    # Move slightly earlier to avoid floating-point boundary overshoots.
    exit_send_deadline = float(np.nextafter(period_end - max_latency, -np.inf))
    entry_cutoff_time = float(np.nextafter(exit_send_deadline - max_latency, -np.inf))

    print("\nValidation — latency comparison")
    print(f"Period end: {period_end:.9f} seconds")
    print(f"Entry cutoff: {entry_cutoff_time:.9f} seconds")
    print(f"Exit submission deadline: {exit_send_deadline:.9f} seconds")

    latency_results = []

    for latency in LATENCY_SCENARIOS:
        scenario_trades, scenario_orders = run_backtest_with_latency(
            events=execution_events,
            scores=validation_scores,
            horizon=HORIZON,
            threshold=SIGNAL_THRESHOLD,
            quantity=TRADE_QUANTITY,
            fee_per_share=FEE_PER_SHARE,
            latency_seconds=latency,
            entry_cutoff_time=entry_cutoff_time,
            exit_send_deadline=exit_send_deadline,
        )

        # Verify that executed trades respect the simulation boundaries.
        if not scenario_trades.empty:
            assert scenario_trades["exit_time"].le(period_end).all()
            assert scenario_trades["exit_index"].lt(val_end).all()
            assert scenario_trades["decision_time"].lt(entry_cutoff_time).all()

        decomposition = decompose_trade_pnl(
            execution_events,
            scenario_trades,
        )

        latency_results.append(
            {
                "latency_ms": latency * 1_000,
                "submitted": len(scenario_orders),
                "rejected": int(scenario_orders["status"].eq("entry_rejected").sum()),
                "trades": len(scenario_trades),
                "forced_exits": int(scenario_trades["forced_exit"].sum()),
                "mid_pnl": decomposition["mid_pnl"].sum(),
                "spread_cost": decomposition["spread_cost"].sum(),
                "fees": scenario_trades["fees"].sum(),
                "net_pnl": scenario_trades["net_pnl"].sum(),
                "mean_net_pnl": scenario_trades["net_pnl"].mean(),
                "win_fraction": (
                    scenario_trades["net_pnl"].gt(0).mean()
                    if not scenario_trades.empty
                    else np.nan
                ),
            }
        )

    latency_summary = pd.DataFrame(latency_results).set_index("latency_ms")

    print(latency_summary.round(4).to_string())
    # 10. Final predictive evaluation on the held-out test set.
    # Reuse models fitted on training data only.
    X_test, y_test = datasets["test"]
    X_test_ofi, y_test_ofi = datasets_ofi["test"]

    test_experiments = [
        ("baseline", baseline, X_test, y_test),
        ("logistic_ofi", logistic_ofi, X_test_ofi, y_test_ofi),
        ("boosting_ofi", boosting_ofi, X_test_ofi, y_test_ofi),
    ]

    test_results = pd.DataFrame.from_dict(
        {name: evaluate_model(model, X, y) for name, model, X, y in test_experiments},
        orient="index",
    )
    test_results.index.name = "model"

    print("\nFinal test results:")
    print(test_results.round(6).to_string())

    comparison = pd.concat(
        {
            "validation": results.loc[test_results.index],
            "test": test_results,
        },
        names=["split"],
    )

    print("\nValidation versus test:")
    print(comparison.round(6).to_string())

    print("\nBoosting + OFI — test classification diagnostics:")
    print_classification_diagnostics(
        boosting_ofi,
        X_test_ofi,
        y_test_ofi,
    )

    # 11. Test backtest with the previously fixed strategy parameters.
    test_probabilities = boosting_ofi.predict_proba(X_test_ofi)
    classes = list(boosting_ofi.classes_)

    test_scores = pd.Series(
        test_probabilities[:, classes.index(1)]
        - test_probabilities[:, classes.index(-1)],
        index=X_test_ofi.index,
        name="score",
    )

    # The test period ends at the last available event.
    test_period_end = float(events["time"].iloc[-1])
    max_latency = max(LATENCY_SCENARIOS)

    # Use identical boundaries across latency scenarios.
    test_exit_deadline = float(np.nextafter(test_period_end - max_latency, -np.inf))
    test_entry_cutoff = float(np.nextafter(test_exit_deadline - max_latency, -np.inf))

    print("\nTest — latency comparison")
    print(f"Signal threshold: {SIGNAL_THRESHOLD}")
    print(f"Quantity: {TRADE_QUANTITY}")
    print(f"Fees per share per transaction: {FEE_PER_SHARE}")
    print(f"Period end: {test_period_end:.9f} seconds")
    print(f"Entry cutoff: {test_entry_cutoff:.9f} seconds")
    print(f"Exit submission deadline: {test_exit_deadline:.9f} seconds")

    test_latency_results = []

    for latency in LATENCY_SCENARIOS:
        test_trades, test_orders = run_backtest_with_latency(
            events=events,
            scores=test_scores,
            horizon=HORIZON,
            threshold=SIGNAL_THRESHOLD,
            quantity=TRADE_QUANTITY,
            fee_per_share=FEE_PER_SHARE,
            latency_seconds=latency,
            entry_cutoff_time=test_entry_cutoff,
            exit_send_deadline=test_exit_deadline,
        )

        if not test_orders.empty:
            assert test_orders["decision_index"].ge(splits["test"].start).all()

        if not test_trades.empty:
            assert test_trades["exit_time"].le(test_period_end).all()
            assert test_trades["exit_index"].lt(len(events)).all()
            assert test_trades["decision_time"].lt(test_entry_cutoff).all()

        test_decomposition = decompose_trade_pnl(
            events,
            test_trades,
        )

        test_latency_results.append(
            {
                "latency_ms": latency * 1_000,
                "submitted": len(test_orders),
                "rejected": int(test_orders["status"].eq("entry_rejected").sum()),
                "trades": len(test_trades),
                "forced_exits": int(test_trades["forced_exit"].sum()),
                "mid_pnl": test_decomposition["mid_pnl"].sum(),
                "spread_cost": test_decomposition["spread_cost"].sum(),
                "fees": test_trades["fees"].sum(),
                "net_pnl": test_trades["net_pnl"].sum(),
                "mean_net_pnl": test_trades["net_pnl"].mean(),
                "win_fraction": (
                    test_trades["net_pnl"].gt(0).mean()
                    if not test_trades.empty
                    else np.nan
                ),
            }
        )

    test_latency_summary = pd.DataFrame(test_latency_results).set_index("latency_ms")

    print(test_latency_summary.round(4).to_string())

    execution_comparison = pd.concat(
        {
            "validation": latency_summary,
            "test": test_latency_summary,
        },
        names=["split"],
    )

    print("\nValidation versus test — execution results:")
    print(execution_comparison.round(4).to_string())
    # Save the current experiment results.
    output_dir = PROJECT_ROOT / "reports" / "baseline_v1"
    output_dir.mkdir(parents=True, exist_ok=True)

    results.to_csv(output_dir / "validation_metrics.csv")
    test_results.to_csv(output_dir / "test_metrics.csv")

    latency_summary.to_csv(output_dir / "validation_latency.csv")
    test_latency_summary.to_csv(output_dir / "test_latency.csv")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
