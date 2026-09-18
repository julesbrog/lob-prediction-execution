"""Named research experiments; existing model and simulator implementations are reused."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import hashlib
import json
import platform
import subprocess
import uuid

import numpy as np
import pandas as pd

from lob.backtest import run_backtest, run_backtest_with_latency
from lob.data import (align_events, compute_horizon_duration, load_messages,
                      load_orderbook, validate_book)
from lob.evaluation import (analyze_aggressive_execution, analyze_score_bins,
                           decompose_trade_pnl, evaluate_model,
                           print_classification_diagnostics)
from lob.features import (build_basic_features, compute_depth_imbalance,
                          compute_event_ofi, compute_rolling_features,
                          compute_rolling_ofi)
from lob.models import fit_baseline, fit_boosting, fit_logistic, fit_mlp
from lob.labels import make_targets
from lob.splits import make_temporal_splits, prepare_datasets
from lob.message_features import (
    compute_activity_features,
    compute_depth_features,
    compute_flow_features,
    compute_noise_feature,
    compute_trade_features,
)
from lob.passive import run_passive_backtest
from lob.uncertainty import block_bootstrap, bootstrap_group_means
from sklearn.inspection import permutation_importance


@dataclass(frozen=True)
class ExperimentConfig:
    project_root: Path
    file_prefix: str = "AAPL_2012-06-21_34200000_57600000"
    n_levels: int = 10
    horizon: int = 50
    epsilon_units: int = 0
    windows: tuple[int, ...] = (10, 50, 100)
    depths: tuple[int, ...] = (5, 10)
    train_fraction: float = 0.6
    val_fraction: float = 0.2
    signal_threshold: float = 0.3
    trade_quantity: int = 1
    fee_per_share: float = 0.0
    latency_scenarios: tuple[float, ...] = (0.0, 0.001, 0.010)
    mlp_seeds: tuple[int, ...] = (0, 1, 2, 3, 42)
    mlp_max_epochs: int = 100
    mlp_patience: int = 10
    # Block bootstrap: blocks of consecutive events or consecutive trades.
    event_block_sizes: tuple[int, ...] = (500, 2_000, 5_000)
    trade_block_size: int = 25
    n_resamples: int = 1_000
    # Message, activity and depth feature families; noise is a control.
    activity_windows: tuple[int, ...] = (50, 100)
    depth_levels: int = 5
    noise_seed: int = 0
    permutation_repeats: int = 5
    # Threshold sweep for the validation-only signal study.
    signal_thresholds: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    # Passive placement: join the best quote (0) or improve inside the spread.
    improve_ticks: tuple[int, ...] = (0, 1)

    @property
    def data_dir(self):
        return self.project_root / "data" / "raw"

    @property
    def feature_columns(self):
        return [
            "spread", "imbalance_1",
            *[f"imbalance_{depth}" for depth in self.depths],
            "weighted_mid_offset",
            *[f"{feature}_{window}" for window in self.windows
              for feature in ("log_return", "realized_vol")],
        ]

    @property
    def ofi_feature_columns(self):
        return self.feature_columns + [f"ofi_{w}" for w in self.windows]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_manifest(config, experiment, output_dir):
    dependencies = {}
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib"):
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            dependencies[package] = None
    root = config.project_root
    files = [root / "run_experiment.py", *sorted((root / "lob").glob("*.py"))]
    sources = {str(p.relative_to(root)): sha256_file(p) for p in files if p.is_file()}
    inputs = {}
    for kind in ("message", "orderbook"):
        p = config.data_dir / f"{config.file_prefix}_{kind}_{config.n_levels}.csv"
        inputs[p.name] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root,
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root,
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        ).strip())
    except (OSError, subprocess.SubprocessError):
        revision, dirty = None, None
    write_json(output_dir / "manifest.json", {
        "experiment": experiment,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": asdict(config),
        "feature_columns": config.feature_columns,
        "ofi_feature_columns": config.ofi_feature_columns,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": dependencies,
        "git_commit": revision, "git_dirty": dirty,
        "source_sha256": sources, "inputs": inputs,
        "protocol": {
            "fit_data": "train only",
            "mlp_selection": "minimum validation log loss; no test evaluation",
            "test": "baseline_v1 reproduces the previously disclosed test experiment",
            "fills": "best level only, no impact, instantaneous acknowledgements",
        },
    })


def save_model_parameters(output_dir, name, model):
    # Source hashes and versions in the manifest complement estimator parameters.
    write_json(output_dir / f"parameters_{name}.json", model.get_params(deep=True))

def load_events(config: ExperimentConfig) -> pd.DataFrame:
    """Load, align and validate one LOBSTER session."""

    messages = load_messages(config.data_dir / f"{config.file_prefix}_message_{config.n_levels}.csv")
    book = load_orderbook(
        config.data_dir / f"{config.file_prefix}_orderbook_{config.n_levels}.csv",
        n_levels=config.n_levels,
    )
    events = align_events(messages, book)

    print(f"Dataset: {config.file_prefix}")
    print(f"Events: {len(events):,} | Book levels: {config.n_levels}")

    checks = validate_book(book, n_levels=config.n_levels)

    print("\nOrder book checks:")
    print(checks.sum().to_string())

    if checks.any().any():
        raise ValueError("Order book flags detected; inspect before proceeding")

    return events


def build_features(events: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    """Build the level-one, depth, rolling and OFI features."""

    features = build_basic_features(events)

    for depth in config.depths:
        features[f"imbalance_{depth}"] = compute_depth_imbalance(
            events,
            n_levels=depth,
        )

    historical = compute_rolling_features(features["mid_price"], windows=config.windows)
    features = features.join(historical)

    event_ofi = compute_event_ofi(events)
    rolling_ofi = compute_rolling_ofi(event_ofi, windows=config.windows)

    if not rolling_ofi.index.equals(features.index):
        raise ValueError("OFI and other features must have the same index")

    features = features.join(rolling_ofi)

    print("\nRolling OFI — missing values:")
    print(rolling_ofi.isna().sum().to_string())

    return features


def prepare_experiment(config: ExperimentConfig) -> dict:
    """Return events, features, targets, splits and the two dataset variants."""

    events = load_events(config)
    features = build_features(events, config)

    targets = make_targets(
        bid_raw=events["bid_price_1"],
        ask_raw=events["ask_price_1"],
        horizon=config.horizon,
        epsilon_units=config.epsilon_units,
    )

    splits = make_temporal_splits(
        n_events=len(events),
        horizon=config.horizon,
        train_fraction=config.train_fraction,
        val_fraction=config.val_fraction,
    )

    datasets = prepare_datasets(features, targets, splits, config.feature_columns)
    datasets_ofi = prepare_datasets(features, targets, splits, config.ofi_feature_columns)

    # Both variants must use identical observations and labels.
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
            X_with_ofi[config.feature_columns].to_numpy(),
        )

    print(f"\nHorizon: {config.horizon} events | Epsilon: ${config.epsilon_units / 20_000:g}")
    print("Prepared datasets:")

    for name in splits:
        X_reference, y_reference = datasets[name]
        X_with_ofi, _ = datasets_ofi[name]

        print(
            f"  {name}: {len(y_reference):,} observations | "
            f"reference: {X_reference.shape[1]} features | "
            f"with OFI: {X_with_ofi.shape[1]} features"
        )

    horizon_seconds = compute_horizon_duration(events, horizon=config.horizon)

    for name in ("train", "validation"):
        X, _ = datasets[name]
        durations = horizon_seconds.loc[X.index]

        print(f"\n{name} — duration of {config.horizon} events, in seconds:")
        print(
            durations.describe(percentiles=[0.01, 0.10, 0.50, 0.90, 0.99]).to_string()
        )

    return {
        "events": events,
        "features": features,
        "targets": targets,
        "splits": splits,
        "datasets": datasets,
        "datasets_ofi": datasets_ofi,
        "horizon_seconds": horizon_seconds,
    }


def compute_scores(model, X: pd.DataFrame) -> pd.Series:
    """Return P(up) - P(down) indexed by original event positions."""

    probabilities = model.predict_proba(X)
    classes = list(model.classes_)

    return pd.Series(
        probabilities[:, classes.index(1)] - probabilities[:, classes.index(-1)],
        index=X.index,
        name="score",
    )


def summarize_trades(trades: pd.DataFrame, label: str) -> None:
    print(f"\n{label}")
    print("Number of trades:", len(trades))

    if trades.empty:
        print("No trades executed.")
        return

    print(trades.head().to_string(index=False))
    print("\nTotal gross PnL ($):", trades["gross_pnl"].sum())
    print("Total fees ($):", trades["fees"].sum())
    print("Total net PnL ($):", trades["net_pnl"].sum())
    print("Mean net PnL per trade ($):", trades["net_pnl"].mean())
    print("Winning trade fraction:", trades["net_pnl"].gt(0).mean())

    print("\nNet PnL by position side (-1: short, +1: long):")
    print(trades.groupby("side")["net_pnl"].agg(["count", "sum", "mean"]).to_string())


def run_latency_scenarios(
    events: pd.DataFrame,
    scores: pd.Series,
    first_decision_index: int,
    label: str,
    config: ExperimentConfig,
    output_dir: Path,
) -> pd.DataFrame:
    """Backtest the same signal under each latency, with shared boundaries."""

    period_end = float(events["time"].iloc[-1])
    max_latency = max(config.latency_scenarios)

    # Move slightly earlier to avoid floating-point boundary overshoots.
    exit_send_deadline = float(np.nextafter(period_end - max_latency, -np.inf))
    entry_cutoff_time = float(np.nextafter(exit_send_deadline - max_latency, -np.inf))

    print(f"\n{label} — latency comparison")
    print(f"Signal threshold: {config.signal_threshold}")
    print(f"Quantity: {config.trade_quantity}")
    print(f"Fees per share per transaction: {config.fee_per_share}")
    print(f"Period end: {period_end:.9f} seconds")
    print(f"Entry cutoff: {entry_cutoff_time:.9f} seconds")
    print(f"Exit submission deadline: {exit_send_deadline:.9f} seconds")

    write_json(output_dir / f"{label.lower()}_execution_boundaries.json", {
        "first_decision_index": first_decision_index,
        "period_end": period_end, "entry_cutoff_time": entry_cutoff_time,
        "exit_send_deadline": exit_send_deadline,
    })
    rows = []

    for latency in config.latency_scenarios:
        trades, orders = run_backtest_with_latency(
            events=events,
            scores=scores,
            horizon=config.horizon,
            threshold=config.signal_threshold,
            quantity=config.trade_quantity,
            fee_per_share=config.fee_per_share,
            latency_seconds=latency,
            entry_cutoff_time=entry_cutoff_time,
            exit_send_deadline=exit_send_deadline,
        )

        # Executed trades must respect the simulation boundaries.
        if not orders.empty:
            assert orders["decision_index"].ge(first_decision_index).all()

        if not trades.empty:
            assert trades["exit_time"].le(period_end).all()
            assert trades["exit_index"].lt(len(events)).all()
            assert trades["decision_time"].lt(entry_cutoff_time).all()

        decomposition = decompose_trade_pnl(events, trades)

        tag = f"{label.lower()}_latency_{latency * 1000:g}ms"
        decomposition.to_csv(output_dir / f"{tag}_trades.csv", index=False)
        orders.to_csv(output_dir / f"{tag}_orders.csv", index=False)

        rows.append(
            {
                "latency_ms": latency * 1_000,
                "submitted": len(orders),
                "rejected": int(orders["status"].eq("entry_rejected").sum()),
                "trades": len(trades),
                "forced_exits": int(trades["forced_exit"].sum()),
                "mid_pnl": decomposition["mid_pnl"].sum(),
                "spread_cost": decomposition["spread_cost"].sum(),
                "fees": trades["fees"].sum(),
                "net_pnl": trades["net_pnl"].sum(),
                "mean_net_pnl": trades["net_pnl"].mean(),
                "win_fraction": (
                    trades["net_pnl"].gt(0).mean() if not trades.empty else np.nan
                ),
            }
        )

    summary = pd.DataFrame(rows).set_index("latency_ms")
    print(summary.round(4).to_string())

    return summary


def experiment_baseline_v1(data: dict, config: ExperimentConfig, output_dir: Path) -> None:
    """Compare logistic and boosting models, reproduce the original held-out experiment."""

    events = data["events"]
    targets = data["targets"]
    splits = data["splits"]
    datasets = data["datasets"]
    datasets_ofi = data["datasets_ofi"]

    X_train, y_train = datasets["train"]
    X_validation, y_validation = datasets["validation"]
    X_train_ofi, y_train_ofi = datasets_ofi["train"]
    X_validation_ofi, y_validation_ofi = datasets_ofi["validation"]

    # 1. Fit all models using training data only.
    baseline = fit_baseline(X_train, y_train)
    logistic_simple = fit_logistic(X_train[["imbalance_1"]], y_train)
    logistic_full = fit_logistic(X_train, y_train)
    logistic_ofi = fit_logistic(X_train_ofi, y_train_ofi)
    boosting_ofi = fit_boosting(X_train_ofi, y_train_ofi)

    for name, model in (
        ("baseline", baseline), ("logistic_simple", logistic_simple),
        ("logistic_full", logistic_full), ("logistic_ofi", logistic_ofi),
        ("boosting_ofi", boosting_ofi),
    ):
        save_model_parameters(output_dir, name, model)

    # 2. Compare all five models on validation.
    experiments = [
        ("baseline", baseline, X_validation, y_validation),
        ("logistic_simple", logistic_simple, X_validation[["imbalance_1"]], y_validation),
        ("logistic_full", logistic_full, X_validation, y_validation),
        ("logistic_ofi", logistic_ofi, X_validation_ofi, y_validation_ofi),
        ("boosting_ofi", boosting_ofi, X_validation_ofi, y_validation_ofi),
    ]

    results = pd.DataFrame.from_dict(
        {name: evaluate_model(model, X, y) for name, model, X, y in experiments},
        orient="index",
    )
    results.index.name = "model"

    print("\nValidation results:")
    print(results.round(6).to_string())

    # 3. Detailed validation diagnostics for the boosting model.
    print("\nGradientBoosting model (14 features) — validation:")
    print_classification_diagnostics(boosting_ofi, X_validation_ofi, y_validation_ofi)

    score_summary = analyze_score_bins(boosting_ofi, X_validation_ofi, targets)

    print("\nGradientBoosting model — future changes by score bin:")
    print(score_summary.round(4).to_string())

    execution_summary = analyze_aggressive_execution(
        boosting_ofi, X_validation_ofi, events, horizon=config.horizon
    )

    print("\nGradientBoosting model — immediate execution diagnostic ($ per share):")
    print(execution_summary.round(4).to_string())

    # 4. Zero-latency validation backtest and PnL decomposition.
    validation_scores = compute_scores(boosting_ofi, X_validation_ofi)

    trades = run_backtest(
        events=events,
        scores=validation_scores,
        horizon=config.horizon,
        threshold=config.signal_threshold,
        quantity=config.trade_quantity,
        fee_per_share=config.fee_per_share,
    )

    summarize_trades(
        trades,
        f"Boosting + OFI — validation backtest (zero latency, fees {config.fee_per_share})",
    )

    decomposed_trades = decompose_trade_pnl(events, trades)

    print("\nValidation — PnL decomposition ($):")
    print(
        decomposed_trades[["mid_pnl", "spread_cost", "fees", "reconstructed_net_pnl"]]
        .sum()
        .round(4)
        .to_string()
    )

    # 5. Latency scenarios on validation, using events strictly before test.
    val_end = splits["test"].start
    execution_events = events.iloc[:val_end].copy()

    latency_summary = run_latency_scenarios(
        execution_events,
        validation_scores,
        first_decision_index=splits["validation"].start,
        label="Validation",
        config=config,
        output_dir=output_dir,
    )

    # 6. Final predictive evaluation on the held-out test set.
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
        {"validation": results.loc[test_results.index], "test": test_results},
        names=["split"],
    )

    print("\nValidation versus test:")
    print(comparison.round(6).to_string())

    print("\nBoosting + OFI — test classification diagnostics:")
    print_classification_diagnostics(boosting_ofi, X_test_ofi, y_test_ofi)

    # 7. Test backtest with the previously fixed strategy parameters.
    test_scores = compute_scores(boosting_ofi, X_test_ofi)

    test_latency_summary = run_latency_scenarios(
        events,
        test_scores,
        first_decision_index=splits["test"].start,
        label="Test",
        config=config,
        output_dir=output_dir,
    )

    execution_comparison = pd.concat(
        {"validation": latency_summary, "test": test_latency_summary},
        names=["split"],
    )

    print("\nValidation versus test — execution results:")
    print(execution_comparison.round(4).to_string())

    # 8. Save the experiment results.

    results.to_csv(output_dir / "validation_metrics.csv")
    test_results.to_csv(output_dir / "test_metrics.csv")
    latency_summary.to_csv(output_dir / "validation_latency.csv")
    test_latency_summary.to_csv(output_dir / "test_latency.csv")
    score_summary.to_csv(output_dir / "validation_score_bins.csv")
    execution_summary.to_csv(output_dir / "validation_execution_bins.csv")
    decomposed_trades.to_csv(output_dir / "validation_trades.csv", index=False)

    print(f"\nResults saved to: {output_dir}")


def experiment_mlp_multiseed(data: dict, config: ExperimentConfig, output_dir: Path) -> None:
    """Train the MLP on the 14 features for several seeds; validation only."""

    datasets_ofi = data["datasets_ofi"]

    X_train_ofi, y_train_ofi = datasets_ofi["train"]
    X_validation_ofi, y_validation_ofi = datasets_ofi["validation"]


    records = []

    for seed in config.mlp_seeds:
        print(f"\n{'=' * 50}")
        print(f"MLP — random_state={seed}")
        print(f"{'=' * 50}")

        mlp_model, history = fit_mlp(
            X_train=X_train_ofi,
            y_train=y_train_ofi,
            X_validation=X_validation_ofi,
            y_validation=y_validation_ofi,
            random_state=seed,
            max_epochs=config.mlp_max_epochs,
            patience=config.mlp_patience,
        )

        save_model_parameters(output_dir, f"mlp_seed_{seed}", mlp_model)

        metrics = evaluate_model(mlp_model, X_validation_ofi, y_validation_ofi)
        best_row = history.loc[history["is_best"]].iloc[0]

        records.append(
            {
                "seed": seed,
                "best_epoch": int(best_row["epoch"]),
                "epochs_run": len(history),
                **metrics,
            }
        )

        history.to_csv(output_dir / f"training_history_seed_{seed}.csv", index=False)

        print(
            f"\nSeed {seed} | best epoch: {int(best_row['epoch'])} | "
            f"validation log loss: {metrics['log_loss']:.6f}"
        )

    seed_results = pd.DataFrame(records).set_index("seed")
    metric_columns = ["log_loss", "accuracy", "macro_f1"]
    summary = seed_results[metric_columns].agg(["mean", "std", "min", "max"])

    print("\nMLP — validation results by seed:")
    print(seed_results.round(6).to_string())

    print("\nMLP — variability across seeds:")
    print(summary.round(6).to_string())

    seed_results.to_csv(output_dir / "validation_metrics_by_seed.csv")
    summary.to_csv(output_dir / "validation_metrics_summary.csv")

    print(f"\nResults saved to: {output_dir}")

def experiment_uncertainty_v1(data: dict, config: ExperimentConfig, output_dir: Path) -> None:
    """Block-bootstrap intervals for the decile means and the backtest PnL.

    Refits the boosting model, then resamples blocks of consecutive events
    (decile analysis) or consecutive trades (backtest) so that temporal
    dependence, including overlapping labels, is preserved.
    """

    events = data["events"]
    targets = data["targets"]
    datasets_ofi = data["datasets_ofi"]

    X_train_ofi, y_train_ofi = datasets_ofi["train"]
    X_validation_ofi, _ = datasets_ofi["validation"]
    X_test_ofi, _ = datasets_ofi["test"]

    boosting_ofi = fit_boosting(X_train_ofi, y_train_ofi)
    save_model_parameters(output_dir, "boosting_ofi", boosting_ofi)

    # 1. Mean future change by score decile on validation, events in order.
    validation_scores = compute_scores(boosting_ofi, X_validation_ofi)

    if not validation_scores.index.is_monotonic_increasing:
        raise ValueError("validation events must be in temporal order")

    future_change_cents = targets.loc[validation_scores.index, "future_change"] * 100
    deciles = pd.Series(
        pd.qcut(validation_scores, q=10, labels=False, duplicates="drop") + 1,
        index=validation_scores.index,
        name="score_decile",
    )

    is_top = deciles.eq(deciles.max()).to_numpy()
    is_bottom = deciles.eq(deciles.min()).to_numpy()
    change = future_change_cents.to_numpy(dtype=float)

    # Rows outside both deciles carry zero weight.
    signed = np.where(is_top, change, np.where(is_bottom, -change, 0.0))
    weights = (is_top | is_bottom).astype(float)
    stacked = np.column_stack([signed, weights])

    decile_tables = []
    spread_rows = []

    for block_size in config.event_block_sizes:
        table = bootstrap_group_means(
            values=future_change_cents,
            groups=deciles,
            block_size=block_size,
            n_resamples=config.n_resamples,
        )
        table.insert(0, "block_size", block_size)
        decile_tables.append(table)

        spread = block_bootstrap(
            stacked,
            lambda sample: float(sample[:, 0].sum() / sample[:, 1].sum()),
            block_size=block_size,
            n_resamples=config.n_resamples,
        )
        spread_rows.append({"block_size": block_size, **spread})

        print(
            f"\nValidation — block size {block_size:,} events: "
            f"top minus bottom decile = {spread['estimate']:.2f} cents "
            f"[{spread['lower']:.2f}, {spread['upper']:.2f}]"
        )

    decile_summary = pd.concat(decile_tables).reset_index()
    reference_block = config.event_block_sizes[len(config.event_block_sizes) // 2]

    print(
        f"\nValidation — mean future change by score decile (cents), "
        f"blocks of {reference_block:,} events, 95% intervals:"
    )
    print(
        decile_summary.loc[decile_summary["block_size"].eq(reference_block)]
        .set_index("score_decile")
        .round(3)
        .to_string()
    )

    decile_summary.to_csv(output_dir / "validation_decile_intervals.csv", index=False)
    pd.DataFrame(spread_rows).to_csv(
        output_dir / "validation_top_minus_bottom.csv", index=False
    )

    # 2. Zero-latency backtest PnL on validation and test, trades in order.
    backtest_rows = []

    for split_name, X in (("validation", X_validation_ofi), ("test", X_test_ofi)):
        scores = compute_scores(boosting_ofi, X)

        trades = run_backtest(
            events=events,
            scores=scores,
            horizon=config.horizon,
            threshold=config.signal_threshold,
            quantity=config.trade_quantity,
            fee_per_share=config.fee_per_share,
        )
        trades = decompose_trade_pnl(events, trades)

        if not trades["entry_index"].is_monotonic_increasing:
            raise ValueError("trades must be in temporal order")

        trades.to_csv(output_dir / f"{split_name}_trades.csv", index=False)

        for column in ("mid_pnl", "spread_cost", "net_pnl"):
            values = trades[column].to_numpy(dtype=float)

            for statistic_name, statistic in (("total", np.sum), ("mean", np.mean)):
                result = block_bootstrap(
                    values,
                    statistic,
                    block_size=min(config.trade_block_size, len(values)),
                    n_resamples=config.n_resamples,
                )
                backtest_rows.append(
                    {
                        "split": split_name,
                        "trades": len(trades),
                        "quantity": column,
                        "statistic": statistic_name,
                        **result,
                    }
                )

    backtest_summary = pd.DataFrame(backtest_rows)

    print(
        f"\nBacktest PnL ($), zero latency, threshold {config.signal_threshold}, "
        f"blocks of {config.trade_block_size} trades, 95% intervals:"
    )
    print(
        backtest_summary.loc[backtest_summary["statistic"].eq("total")]
        .set_index(["split", "quantity"])
        .drop(columns="statistic")
        .round(3)
        .to_string()
    )

    backtest_summary.to_csv(output_dir / "backtest_intervals.csv", index=False)

    print(f"\nResults saved to: {output_dir}")


def build_feature_families(data: dict, config: ExperimentConfig) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Join message, activity, depth and noise features to the book features."""

    events = data["events"]
    features = data["features"]

    extra = [
        compute_trade_features(events, windows=config.windows),
        compute_flow_features(events, windows=config.windows),
        compute_activity_features(events, windows=config.activity_windows),
        compute_depth_features(events, n_levels=config.depth_levels),
        compute_noise_feature(events.index, random_state=config.noise_seed).to_frame(),
    ]

    for frame in extra:
        if not frame.index.equals(features.index):
            raise ValueError("all feature frames must share the event index")

        overlap = set(frame.columns) & set(features.columns)
        if overlap:
            raise ValueError(f"duplicate feature names: {sorted(overlap)}")

        features = features.join(frame)

    families = {
        "book": list(config.feature_columns),
        "ofi": [f"ofi_{w}" for w in config.windows],
        "trades": [
            f"{name}_{w}"
            for w in config.windows
            for name in ("signed_trade_volume", "trade_volume")
        ],
        "flow": [
            f"{name}_{w}"
            for w in config.windows
            for name in ("net_order_flow", "cancel_volume")
        ],
        "activity": ["log_time_gap"]
        + [f"event_rate_{w}" for w in config.activity_windows],
        "depth": [
            "log_bid_size_1",
            "log_ask_size_1",
            f"log_bid_depth_{config.depth_levels}",
            f"log_ask_depth_{config.depth_levels}",
        ],
        "noise": ["noise"],
    }

    return features, families


def experiment_features_v1(data: dict, config: ExperimentConfig, output_dir: Path) -> None:
    """Add feature families one at a time, ablate them, and rank by permutation.

    Every feature set is evaluated on the same validation rows. The noise
    column is a control: a family that does not beat it adds nothing.
    """

    targets = data["targets"]
    splits = data["splits"]

    features, families = build_feature_families(data, config)
    all_columns = [col for family in families.values() for col in family]

    # One shared row mask so every feature set sees identical observations.
    datasets = prepare_datasets(features, targets, splits, all_columns)
    X_train, y_train = datasets["train"]
    X_validation, y_validation = datasets["validation"]

    print(
        f"\nShared rows: {len(y_train):,} train | {len(y_validation):,} validation "
        f"| {len(all_columns)} candidate features"
    )

    reference = families["book"] + families["ofi"]
    real_families = [name for name in families if name not in ("book", "ofi", "noise")]
    full = reference + [col for name in real_families for col in families[name]]

    feature_sets = {"reference": reference}
    for name in real_families + ["noise"]:
        feature_sets[f"reference+{name}"] = reference + families[name]
    feature_sets["all"] = full
    for name in ["ofi"] + real_families:
        feature_sets[f"all-{name}"] = [col for col in full if col not in families[name]]
    feature_sets["all+noise"] = full + families["noise"]

    write_json(output_dir / "feature_sets.json", feature_sets)
    write_json(output_dir / "feature_families.json", families)

    rows = []
    fitted = {}

    for set_name, columns in feature_sets.items():
        for model_name, fit in (("logistic", fit_logistic), ("boosting", fit_boosting)):
            model = fit(X_train[columns], y_train)
            metrics = evaluate_model(model, X_validation[columns], y_validation)
            fitted[(set_name, model_name)] = model

            rows.append(
                {
                    "feature_set": set_name,
                    "model": model_name,
                    "n_features": len(columns),
                    **metrics,
                }
            )

            print(
                f"{set_name:<20} {model_name:<9} {len(columns):>3} features | "
                f"log loss {metrics['log_loss']:.6f} | "
                f"accuracy {metrics['accuracy']:.4f}",
                flush=True,
            )

    results = pd.DataFrame(rows)

    # Gain relative to the 14-feature reference, per model.
    reference_loss = results.loc[results["feature_set"].eq("reference")].set_index("model")["log_loss"]
    results["log_loss_vs_reference"] = results["log_loss"] - results["model"].map(reference_loss)

    full_loss = results.loc[results["feature_set"].eq("all")].set_index("model")["log_loss"]
    results["log_loss_vs_all"] = results["log_loss"] - results["model"].map(full_loss)

    print("\nValidation log loss by feature set:")
    print(
        results.pivot(index="feature_set", columns="model", values="log_loss")
        .loc[list(feature_sets)]
        .round(6)
        .to_string()
    )

    results.to_csv(output_dir / "validation_metrics_by_feature_set.csv", index=False)

    # Permutation importance of the boosting model with the noise control.
    columns = feature_sets["all+noise"]
    model = fitted[("all+noise", "boosting")]

    importance = permutation_importance(
        model,
        X_validation[columns],
        y_validation,
        scoring="neg_log_loss",
        n_repeats=config.permutation_repeats,
        random_state=config.noise_seed,
        n_jobs=1,
    )

    column_family = {col: name for name, cols in families.items() for col in cols}

    importance_table = pd.DataFrame(
        {
            "feature": columns,
            "family": [column_family[col] for col in columns],
            "log_loss_increase": importance.importances_mean,
            "std": importance.importances_std,
        }
    ).sort_values("log_loss_increase", ascending=False)

    noise_level = float(
        importance_table.loc[importance_table["feature"].eq("noise"), "log_loss_increase"].iloc[0]
    )
    importance_table["above_noise"] = importance_table["log_loss_increase"] > noise_level

    print(
        f"\nPermutation importance on validation (boosting, all+noise, "
        f"{config.permutation_repeats} repeats), increase in log loss:"
    )
    print(importance_table.round(6).to_string(index=False))

    importance_table.to_csv(output_dir / "permutation_importance.csv", index=False)

    family_importance = (
        importance_table.groupby("family")["log_loss_increase"]
        .agg(["sum", "mean", "max", "count"])
        .sort_values("sum", ascending=False)
    )

    print("\nPermutation importance by family:")
    print(family_importance.round(6).to_string())

    family_importance.to_csv(output_dir / "permutation_importance_by_family.csv")

    print(f"\nResults saved to: {output_dir}")


def experiment_signal_v1(data: dict, config: ExperimentConfig, output_dir: Path) -> None:
    """Does the 33-feature model move the signal in cents? Validation only.

    For the 14-feature reference and the full feature set: decile means of
    the future mid-price change with block bootstrap intervals, aggressive
    round-trip PnL by decile, and a zero-latency backtest across thresholds.
    """

    events = data["events"]
    targets = data["targets"]
    splits = data["splits"]

    features, families = build_feature_families(data, config)
    reference = families["book"] + families["ofi"]
    full = reference + [
        col for name in families if name not in ("book", "ofi", "noise")
        for col in families[name]
    ]
    feature_sets = {"reference": reference, "all": full}

    datasets = prepare_datasets(features, targets, splits, full)
    X_train, y_train = datasets["train"]
    X_validation, _ = datasets["validation"]

    block_size = config.event_block_sizes[len(config.event_block_sizes) // 2]

    decile_rows = []
    extreme_rows = []
    execution_rows = []
    threshold_rows = []

    for set_name, columns in feature_sets.items():
        model = fit_boosting(X_train[columns], y_train)
        save_model_parameters(output_dir, f"boosting_{set_name}", model)

        scores = compute_scores(model, X_validation[columns])
        change_cents = targets.loc[scores.index, "future_change"] * 100

        deciles = pd.Series(
            pd.qcut(scores, q=10, labels=False, duplicates="drop") + 1,
            index=scores.index,
            name="score_decile",
        )

        # 1. Decile means with intervals.
        table = bootstrap_group_means(
            values=change_cents,
            groups=deciles,
            block_size=block_size,
            n_resamples=config.n_resamples,
        )
        table.insert(0, "feature_set", set_name)
        decile_rows.append(table.reset_index())

        is_top = deciles.eq(deciles.max()).to_numpy()
        is_bottom = deciles.eq(deciles.min()).to_numpy()
        change = change_cents.to_numpy(dtype=float)
        stacked = np.column_stack([
            np.where(is_top, change, np.where(is_bottom, -change, 0.0)),
            (is_top | is_bottom).astype(float),
        ])
        extreme = block_bootstrap(
            stacked,
            lambda sample: float(sample[:, 0].sum() / sample[:, 1].sum()),
            block_size=block_size,
            n_resamples=config.n_resamples,
        )
        extreme_rows.append({"feature_set": set_name, **extreme})

        print(
            f"\n{set_name} ({len(columns)} features) — favourable move in the "
            f"extreme deciles: {extreme['estimate']:.2f} cents "
            f"[{extreme['lower']:.2f}, {extreme['upper']:.2f}]"
        )

        # 2. Aggressive round trips by decile (overlapping, descriptive).
        execution = analyze_aggressive_execution(
            model, X_validation[columns], events, horizon=config.horizon
        )
        execution.insert(0, "feature_set", set_name)
        execution_rows.append(execution.reset_index())

        # 3. Zero-latency backtest across thresholds.
        for threshold in config.signal_thresholds:
            trades = run_backtest(
                events=events,
                scores=scores,
                horizon=config.horizon,
                threshold=threshold,
                quantity=config.trade_quantity,
                fee_per_share=config.fee_per_share,
            )
            trades = decompose_trade_pnl(events, trades)

            row = {
                "feature_set": set_name,
                "threshold": threshold,
                "trades": len(trades),
                "mid_pnl_per_trade": trades["mid_pnl"].mean() if len(trades) else np.nan,
                "spread_cost_per_trade": trades["spread_cost"].mean() if len(trades) else np.nan,
                "net_pnl_per_trade": trades["net_pnl"].mean() if len(trades) else np.nan,
                "net_pnl_total": trades["net_pnl"].sum(),
            }

            if len(trades) >= 2 * config.trade_block_size:
                interval = block_bootstrap(
                    trades["mid_pnl"].to_numpy(dtype=float),
                    np.mean,
                    block_size=config.trade_block_size,
                    n_resamples=config.n_resamples,
                )
                row["mid_pnl_per_trade_lower"] = interval["lower"]
                row["mid_pnl_per_trade_upper"] = interval["upper"]

            threshold_rows.append(row)

    decile_summary = pd.concat(decile_rows, ignore_index=True)
    extreme_summary = pd.DataFrame(extreme_rows)
    execution_summary = pd.concat(execution_rows, ignore_index=True)
    threshold_summary = pd.DataFrame(threshold_rows)

    print(f"\nValidation — mean future change by decile (cents), 95% intervals, {block_size:,}-event blocks:")
    print(
        decile_summary.pivot(index="score_decile", columns="feature_set", values="estimate")
        .round(3)
        .to_string()
    )

    print("\nValidation — aggressive round trips by decile (cents per share):")
    view = execution_summary.assign(
        long=lambda d: d["mean_long_pnl"] * 100,
        short=lambda d: d["mean_short_pnl"] * 100,
    )
    print(
        view.pivot(index="score_bin", columns="feature_set", values=["long", "short"])
        .round(2)
        .to_string()
    )

    print("\nValidation — zero-latency backtest by threshold ($ per trade):")
    print(threshold_summary.round(4).to_string(index=False))

    decile_summary.to_csv(output_dir / "validation_decile_intervals.csv", index=False)
    extreme_summary.to_csv(output_dir / "validation_extreme_deciles.csv", index=False)
    execution_summary.to_csv(output_dir / "validation_execution_bins.csv", index=False)
    threshold_summary.to_csv(output_dir / "validation_threshold_sweep.csv", index=False)

    print(f"\nResults saved to: {output_dir}")


def experiment_passive_v1(data: dict, config: ExperimentConfig, output_dir: Path) -> None:
    """Passive versus aggressive execution of the same signals, validation only.

    For each feature set, threshold and placement (join the queue or improve
    by one tick): fill rate, PnL decomposition of filled orders, the
    mid-price move conditional on being filled or not (adverse selection),
    and the aggressive strategy on the same decisions for comparison.
    """

    events = data["events"]
    targets = data["targets"]
    splits = data["splits"]

    features, families = build_feature_families(data, config)
    reference = families["book"] + families["ofi"]
    full = reference + [
        col for name in families if name not in ("book", "ofi", "noise")
        for col in families[name]
    ]
    feature_sets = {"reference": reference, "all": full}

    datasets = prepare_datasets(features, targets, splits, full)
    X_train, y_train = datasets["train"]
    X_validation, _ = datasets["validation"]

    summary_rows = []
    all_orders = []

    for set_name, columns in feature_sets.items():
        model = fit_boosting(X_train[columns], y_train)
        save_model_parameters(output_dir, f"boosting_{set_name}", model)
        scores = compute_scores(model, X_validation[columns])

        for threshold in config.signal_thresholds:
            aggressive = decompose_trade_pnl(
                events,
                run_backtest(
                    events=events, scores=scores, horizon=config.horizon,
                    threshold=threshold, quantity=config.trade_quantity,
                    fee_per_share=config.fee_per_share,
                ),
            )

            for improve_ticks in config.improve_ticks:
                orders = run_passive_backtest(
                    events=events, scores=scores, horizon=config.horizon,
                    threshold=threshold, improve_ticks=improve_ticks,
                    quantity=config.trade_quantity,
                    fee_per_share=config.fee_per_share,
                )
                orders.insert(0, "feature_set", set_name)
                orders.insert(1, "threshold", threshold)
                all_orders.append(orders)

                filled = orders.loc[orders["filled"]]
                unfilled = orders.loc[~orders["filled"]]

                row = {
                    "feature_set": set_name,
                    "threshold": threshold,
                    "improve_ticks": improve_ticks,
                    "orders": len(orders),
                    "filled": len(filled),
                    "fill_rate": len(filled) / len(orders) if len(orders) else np.nan,
                    "median_fill_delay_events": (
                        (filled["fill_index"] - filled["decision_index"]).median()
                        if len(filled) else np.nan
                    ),
                    "median_fill_delay_seconds": (
                        (filled["fill_time"] - filled["decision_time"]).median()
                        if len(filled) else np.nan
                    ),
                    "mid_move_filled": filled["mid_move"].mean() if len(filled) else np.nan,
                    "mid_move_unfilled": unfilled["mid_move"].mean() if len(unfilled) else np.nan,
                    "mid_move_all_orders": orders["mid_move"].mean() if len(orders) else np.nan,
                    "entry_edge_per_fill": filled["entry_edge"].mean() if len(filled) else np.nan,
                    "exit_cost_per_fill": filled["exit_cost"].mean() if len(filled) else np.nan,
                    "net_pnl_per_fill": filled["net_pnl"].mean() if len(filled) else np.nan,
                    "net_pnl_per_order": filled["net_pnl"].sum() / len(orders) if len(orders) else np.nan,
                    "passive_net_total": filled["net_pnl"].sum(),
                    "aggressive_trades": len(aggressive),
                    "aggressive_net_per_trade": aggressive["net_pnl"].mean() if len(aggressive) else np.nan,
                    "aggressive_net_total": aggressive["net_pnl"].sum(),
                }

                # Interval on net PnL per order (unfilled orders count as zero),
                # blocks of consecutive orders.
                per_order = orders["net_pnl"].fillna(0.0).to_numpy(dtype=float)
                if len(per_order) >= 2 * config.trade_block_size:
                    interval = block_bootstrap(
                        per_order, np.mean,
                        block_size=config.trade_block_size,
                        n_resamples=config.n_resamples,
                    )
                    row["net_pnl_per_order_lower"] = interval["lower"]
                    row["net_pnl_per_order_upper"] = interval["upper"]

                summary_rows.append(row)

                print(
                    f"{set_name:<10} thr {threshold:.1f} improve {improve_ticks} | "
                    f"orders {len(orders):>5} fill {row['fill_rate']:.2f} | "
                    f"net/fill {row['net_pnl_per_fill']:+.4f} "
                    f"net/order {row['net_pnl_per_order']:+.4f} | "
                    f"aggressive net/trade {row['aggressive_net_per_trade']:+.4f}",
                    flush=True,
                )

    summary = pd.DataFrame(summary_rows)
    orders_table = pd.concat(all_orders, ignore_index=True)

    print("\nValidation — passive execution summary ($ per share unless stated):")
    print(
        summary[[
            "feature_set", "threshold", "improve_ticks", "orders", "fill_rate",
            "median_fill_delay_seconds", "mid_move_filled", "mid_move_unfilled",
            "entry_edge_per_fill", "exit_cost_per_fill", "net_pnl_per_fill",
            "net_pnl_per_order", "aggressive_net_per_trade",
        ]].round(4).to_string(index=False)
    )

    summary.to_csv(output_dir / "validation_passive_summary.csv", index=False)
    orders_table.to_csv(output_dir / "validation_passive_orders.csv", index=False)

    print(f"\nResults saved to: {output_dir}")


EXPERIMENTS = {
    "baseline_v1": experiment_baseline_v1,
    "mlp_multiseed": experiment_mlp_multiseed,
    "uncertainty_v1": experiment_uncertainty_v1,
    "features_v1": experiment_features_v1,
    "signal_v1": experiment_signal_v1,
    "passive_v1": experiment_passive_v1,
}


def run_experiment(experiment: str, config: ExperimentConfig, output_dir=None):
    """Create a new result directory; never overwrite an existing run."""
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment}")
    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        ticker = config.file_prefix.split("_")[0]
        output_dir = config.project_root / "reports" / "runs" / (
            f"{experiment}_{ticker}_{stamp}_{uuid.uuid4().hex[:8]}"
        )
    output_dir = Path(output_dir).resolve()
    # This fails before loading data or fitting if the destination exists.
    output_dir.mkdir(parents=True, exist_ok=False)
    status_path = output_dir / "status.json"
    write_json(status_path, {"status": "running"})
    print(f"Output directory: {output_dir}", flush=True)
    try:
        save_manifest(config, experiment, output_dir)
        data = prepare_experiment(config)
        write_json(output_dir / "dataset_summary.json", {
            name: {
                "slice_start": selection.start, "slice_stop": selection.stop,
                "usable_rows": len(data["datasets_ofi"][name][0]),
                "first_event": int(data["datasets_ofi"][name][0].index[0]),
                "last_event": int(data["datasets_ofi"][name][0].index[-1]),
            }
            for name, selection in data["splits"].items()
        })
        EXPERIMENTS[experiment](data, config, output_dir)
    except BaseException as error:
        write_json(status_path, {"status": "failed", "error": str(error),
                                 "error_type": type(error).__name__})
        raise
    write_json(status_path, {"status": "completed"})
    return output_dir
