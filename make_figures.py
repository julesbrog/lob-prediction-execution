"""Build the README figures from the saved CSV reports.

Usage:
    python make_figures.py

Defaults to the saved reference CSVs; directories can be selected explicitly.
Only figures are regenerated. Raw data and model fitting are not required.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
GRAY = "#8a8985"
TEXT = "#0b0b0b"

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6e5e2",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": TEXT,
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
    }
)


def plot_model_comparison(baseline_dir: Path, mlp_dir: Path, output_dir: Path) -> None:
    validation = pd.read_csv(baseline_dir / "validation_metrics.csv")
    test = pd.read_csv(baseline_dir / "test_metrics.csv")
    mlp = pd.read_csv(mlp_dir / "validation_metrics_summary.csv")

    validation = validation.set_index("model")["log_loss"]
    test = test.set_index("model")["log_loss"]

    mlp_mean = float(mlp.set_index(mlp.columns[0]).loc["mean", "log_loss"])
    mlp_std = float(mlp.set_index(mlp.columns[0]).loc["std", "log_loss"])

    order = [
        "baseline",
        "logistic_simple",
        "logistic_full",
        "logistic_ofi",
        "boosting_ofi",
        "mlp_ofi",
    ]
    labels = [
        "prior",
        "logistic\nimbalance only",
        "logistic\n11 features",
        "logistic\n14 features",
        "boosting\n14 features",
        "MLP\n14 features",
    ]

    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = list(range(len(order)))

    val_values = [mlp_mean if m == "mlp_ofi" else validation.loc[m] for m in order]
    test_values = [test.get(m, float("nan")) for m in order]

    ax.plot(x, val_values, "o", color=BLUE, markersize=7, label="validation")
    ax.plot(
        x,
        test_values,
        "s",
        color=ORANGE,
        markersize=7,
        label="held-out test — baseline experiment",
    )
    ax.errorbar(
        x[-1],
        mlp_mean,
        yerr=mlp_std,
        color=BLUE,
        capsize=4,
        linewidth=1.2,
    )
    fig.text(
        0.99, 0.015,
        "MLP: mean ± seed std; not a confidence interval",
        ha="right", fontsize=7, color="#52514e",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.set_ylabel("log loss (lower is better)")
    ax.set_title("Three-class direction prediction", loc="left")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="x", visible=False)

    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(output_dir / "model_comparison.png")
    plt.close(fig)


def plot_score_bins(
    baseline_dir: Path, output_dir: Path, uncertainty_dir: Path | None = None
) -> None:
    bins = pd.read_csv(baseline_dir / "validation_score_bins.csv")
    execution = pd.read_csv(
        baseline_dir / "validation_execution_bins.csv"
    )

    if bins["score_bin"].duplicated().any() or execution["score_bin"].duplicated().any():
        raise ValueError("Score bin IDs must be unique")
    bins = bins.sort_values("score_bin")
    execution = execution.set_index("score_bin")
    if set(bins["score_bin"]) != set(execution.index):
        raise ValueError("Prediction and execution bins must match")
    execution = execution.loc[bins["score_bin"]]
    if not np.array_equal(bins["count"].to_numpy(), execution["count"].to_numpy()):
        raise ValueError("Prediction and execution bin counts must match")

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(8, 3.2))

    deciles = bins["score_bin"] + 1

    errors = None
    if uncertainty_dir is not None:
        intervals = pd.read_csv(uncertainty_dir / "validation_decile_intervals.csv")
        block = sorted(intervals["block_size"].unique())
        block = block[len(block) // 2]
        intervals = intervals.loc[intervals["block_size"].eq(block)]
        intervals = intervals.set_index("score_decile").loc[deciles]
        if not np.allclose(intervals["estimate"], bins["mean_change_dollars"] * 100, atol=0.02):
            raise ValueError("Decile intervals do not match the score bins")
        errors = np.vstack([
            intervals["estimate"] - intervals["lower"],
            intervals["upper"] - intervals["estimate"],
        ])
        ax_left.set_title(
            f"Validation: future mid-price change\n95% block bootstrap, {block:,}-event blocks",
            loc="left", fontsize=9,
        )

    ax_left.bar(
        deciles, bins["mean_change_dollars"] * 100, color=BLUE, width=0.7,
        yerr=errors, capsize=2, ecolor=TEXT, error_kw={"linewidth": 0.8},
    )
    ax_left.axhline(0, color=GRAY, linewidth=0.8)
    ax_left.set_xlabel("score decile (P(up) - P(down))")
    ax_left.set_ylabel("mean future mid-price change (cents)")
    if errors is None:
        ax_left.set_title("Validation: future mid-price change", loc="left")
    ax_left.grid(axis="x", visible=False)

    ax_right.plot(
        deciles,
        execution["mean_long_pnl"] * 100,
        color=AQUA,
        marker="o",
        markersize=4,
        label="buy at ask, sell at bid",
    )
    ax_right.plot(
        deciles,
        execution["mean_short_pnl"] * 100,
        color=ORANGE,
        marker="o",
        markersize=4,
        label="sell at bid, buy at ask",
    )
    ax_right.plot(
        deciles,
        -execution["mean_spread"] * 100,
        color=GRAY,
        linestyle="--",
        label="negative mean entry spread (reference)",
    )
    ax_right.axhline(0, color=GRAY, linewidth=0.8)
    ax_right.set_xlabel("score decile")
    ax_right.set_ylabel("mean round-trip PnL per share (cents)")
    ax_right.set_title("Hypothetical round trips, before fees", loc="left")
    ax_right.legend(frameon=False, fontsize=8)
    ax_right.grid(axis="x", visible=False)

    fig.tight_layout()
    fig.savefig(output_dir / "score_bins.png")
    plt.close(fig)


def plot_pnl_decomposition(baseline_dir: Path, output_dir: Path) -> None:
    validation = pd.read_csv(baseline_dir / "validation_latency.csv")
    test = pd.read_csv(baseline_dir / "test_latency.csv")

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)

    for ax, frame, title in (
        (axes[0], validation, "validation"),
        (axes[1], test, "test"),
    ):
        x = range(len(frame))
        width = 0.26

        ax.bar(
            [i - width for i in x],
            frame["mid_pnl"],
            width,
            color=AQUA,
            label="mid-price PnL",
        )
        ax.bar(
            list(x),
            -frame["spread_cost"],
            width,
            color=ORANGE,
            label="spread cost",
        )
        ax.bar(
            [i + width for i in x],
            frame["net_pnl"],
            width,
            color=GRAY,
            label="net PnL",
        )

        ax.axhline(0, color=GRAY, linewidth=0.8)
        ax.set_xticks(list(x))
        ax.set_xticklabels(
            [
                f"{int(ms)} ms\n{int(n)} trades"
                for ms, n in zip(frame["latency_ms"], frame["trades"])
            ]
        )
        ax.set_title(f"{title} — aggressive execution", loc="left")
        ax.grid(axis="x", visible=False)

    if any(not np.allclose(frame["fees"], 0) for frame in (validation, test)):
        fig.suptitle("Net PnL includes fees (fees not shown separately)", fontsize=9)

    axes[0].set_ylabel("dollars over the period")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8,
               loc="lower center", ncol=3)

    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    fig.savefig(output_dir / "pnl_decomposition.png")
    plt.close(fig)


def plot_mlp_training(mlp_dir: Path, output_dir: Path) -> None:
    directory = mlp_dir
    histories = sorted(directory.glob("training_history_seed_*.csv"))

    if not histories:
        raise FileNotFoundError(f"No MLP histories found in {directory}")

    fig, ax = plt.subplots(figsize=(6, 3.2))

    for i, path in enumerate(histories):
        history = pd.read_csv(path)
        ax.plot(
            history["epoch"],
            history["validation_log_loss"],
            color=BLUE,
            alpha=0.7,
            linewidth=1.2,
            label="validation" if i == 0 else None,
        )
        ax.plot(
            history["epoch"],
            history["train_log_loss"],
            color=ORANGE,
            alpha=0.7,
            linewidth=1.2,
            label="train" if i == 0 else None,
        )

        selected = history["is_best"].astype(str).str.lower()
        if not selected.isin(["true", "false"]).all() or selected.eq("true").sum() != 1:
            raise ValueError(f"Expected exactly one selected epoch in {path}")
        best = history.loc[selected.eq("true")]
        ax.scatter(
            best["epoch"],
            best["validation_log_loss"],
            color=BLUE,
            s=18,
            zorder=3,
        )

    ax.set_xlabel("epoch")
    ax.set_ylabel("log loss")
    ax.set_title("MLP learning curves (dots: selected validation epoch)", loc="left")
    ax.legend(frameon=False)
    ax.grid(axis="x", visible=False)

    fig.tight_layout()
    fig.savefig(output_dir / "mlp_training.png")
    plt.close(fig)


def plot_feature_families(features_dir: Path, output_dir: Path) -> None:
    """Ablation by family (left) and permutation importance (right)."""

    metrics = pd.read_csv(features_dir / "validation_metrics_by_feature_set.csv")
    importance = pd.read_csv(features_dir / "permutation_importance.csv")

    boosting = metrics.loc[metrics["model"].eq("boosting")].set_index("feature_set")
    logistic = metrics.loc[metrics["model"].eq("logistic")].set_index("feature_set")

    order = [
        "reference", "reference+noise", "reference+depth", "reference+flow",
        "reference+activity", "reference+trades", "all",
        "all-flow", "all-activity", "all-depth", "all-ofi", "all-trades",
    ]
    missing = [name for name in order if name not in boosting.index]
    if missing:
        raise ValueError(f"Missing feature sets: {missing}")

    labels = [name.replace("reference", "ref").replace("+", " + ").replace("-", " − ")
              for name in order]

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(9, 4.2), gridspec_kw={"width_ratios": [1, 1.15]}
    )

    y = np.arange(len(order))
    ax_left.plot(logistic.loc[order, "log_loss"], y, "s", color=ORANGE,
                 markersize=6, label="logistic")
    ax_left.plot(boosting.loc[order, "log_loss"], y, "o", color=BLUE,
                 markersize=6, label="boosting")
    ax_left.axvline(boosting.loc["reference", "log_loss"], color=GRAY,
                    linestyle="--", linewidth=0.8)
    ax_left.axhline(6.5, color="#e6e5e2", linewidth=0.8)
    ax_left.set_yticks(y)
    ax_left.set_yticklabels(labels)
    ax_left.invert_yaxis()
    ax_left.set_xlabel("validation log loss")
    ax_left.set_title("Adding a family to the reference (top),\nremoving one from the full set (bottom)",
                      loc="left", fontsize=9)
    ax_left.legend(frameon=False, fontsize=8, loc="lower left")
    ax_left.grid(axis="y", visible=False)

    top = importance.sort_values("log_loss_increase", ascending=False).head(15)
    noise = float(importance.loc[importance["feature"].eq("noise"), "log_loss_increase"].iloc[0])
    family_colors = {
        "trades": BLUE, "ofi": AQUA, "book": GRAY, "depth": ORANGE,
        "activity": "#4a3aa7", "flow": "#e87ba4", "noise": TEXT,
    }
    y2 = np.arange(len(top))
    ax_right.barh(
        y2, top["log_loss_increase"] * 1000,
        xerr=top["std"] * 1000, color=[family_colors[f] for f in top["family"]],
        height=0.7, error_kw={"linewidth": 0.8}, ecolor=TEXT,
    )
    ax_right.axvline(noise * 1000, color=TEXT, linestyle=":", linewidth=0.8)
    ax_right.set_yticks(y2)
    ax_right.set_yticklabels(top["feature"], fontsize=8)
    ax_right.invert_yaxis()
    ax_right.set_xlabel("increase in validation log loss when permuted (×1000)")
    ax_right.set_title("Permutation importance, boosting on all features\n(dotted: noise control)",
                       loc="left", fontsize=9)
    ax_right.grid(axis="y", visible=False)

    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=f) for f, c in family_colors.items()
               if f in set(top["family"])]
    ax_right.legend(handles=handles, frameon=False, fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(output_dir / "feature_families.png")
    plt.close(fig)


def plot_passive_execution(passive_dir: Path, output_dir: Path) -> None:
    """Fill rate, adverse selection and net PnL per order across thresholds."""

    summary = pd.read_csv(passive_dir / "validation_passive_summary.csv")
    frame = summary.loc[summary["feature_set"].eq("all")]
    if frame.empty:
        raise ValueError("passive summary has no rows for the full feature set")

    join = frame.loc[frame["improve_ticks"].eq(0)].sort_values("threshold")
    improve = frame.loc[frame["improve_ticks"].eq(1)].sort_values("threshold")

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3))

    ax = axes[0]
    ax.plot(join["threshold"], join["fill_rate"] * 100, "o-", color=BLUE,
            markersize=4, label="join the queue")
    ax.plot(improve["threshold"], improve["fill_rate"] * 100, "s-", color=ORANGE,
            markersize=4, label="improve one tick")
    ax.set_ylim(0, None)
    ax.set_xlabel("score threshold")
    ax.set_ylabel("orders filled within 50 events (%)")
    ax.set_title("Fill rate", loc="left")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    for frame_, marker, colour, label in (
        (join, "o", BLUE, "join"), (improve, "s", ORANGE, "improve"),
    ):
        ax.plot(frame_["threshold"], frame_["mid_move_filled"] * 100, marker + "-",
                color=colour, markersize=4, label=f"{label}: filled")
        ax.plot(frame_["threshold"], frame_["mid_move_unfilled"] * 100, marker + "--",
                color=colour, markersize=4, alpha=0.6, label=f"{label}: not filled")
    ax.axhline(0, color=GRAY, linewidth=0.8)
    ax.set_xlabel("score threshold")
    ax.set_ylabel("mid-price move in our favour (cents)")
    ax.set_title("Adverse selection", loc="left")
    ax.legend(frameon=False, fontsize=7, ncol=2)

    ax = axes[2]
    ax.plot(join["threshold"], join["net_pnl_per_order"] * 100, "o-", color=BLUE,
            markersize=4, label="passive, join")
    ax.plot(improve["threshold"], improve["net_pnl_per_order"] * 100, "s-",
            color=ORANGE, markersize=4, label="passive, improve")
    ax.plot(join["threshold"], join["aggressive_net_per_trade"] * 100, "^-",
            color=GRAY, markersize=4, label="aggressive")
    ax.axhline(0, color=GRAY, linewidth=0.8)
    ax.set_xlabel("score threshold")
    ax.set_ylabel("net PnL per decision (cents)")
    ax.set_title("Net PnL per decision", loc="left")
    ax.legend(frameon=False, fontsize=8)

    for ax in axes:
        ax.grid(axis="x", visible=False)

    fig.suptitle("Validation, boosting on 33 features, one share, zero latency",
                 fontsize=9, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(output_dir / "passive_execution.png")
    plt.close(fig)


TICKER_DIRS = {
    "AAPL": "baseline_v1", "AMZN": "baseline_AMZN", "GOOG": "baseline_GOOG",
    "INTC": "baseline_INTC", "MSFT": "baseline_MSFT",
}


def summarize_tickers(reports_dir: Path) -> pd.DataFrame:
    """One row per ticker from the saved baseline runs."""

    rows = []
    for ticker, folder in TICKER_DIRS.items():
        directory = reports_dir / folder
        if not directory.is_dir():
            continue
        validation = pd.read_csv(directory / "validation_metrics.csv").set_index("model")
        test = pd.read_csv(directory / "test_metrics.csv").set_index("model")
        bins = pd.read_csv(directory / "validation_score_bins.csv")
        execution = pd.read_csv(directory / "validation_execution_bins.csv")
        latency = pd.read_csv(directory / "test_latency.csv").set_index("latency_ms")
        zero = latency.loc[0.0]
        spread = float((execution["mean_spread"] * execution["count"]).sum() / execution["count"].sum())
        top = bins.sort_values("score_bin").iloc[-1]
        bottom = bins.sort_values("score_bin").iloc[0]
        rows.append({
            "ticker": ticker,
            "events": int(bins["count"].sum() * 5),  # validation is 20% of the session
            "spread_cents": spread * 100,
            "flat_share": float((bins["flat_fraction"] * bins["count"]).sum() / bins["count"].sum()),
            "prior_val": validation.loc["baseline", "log_loss"],
            "logistic_val": validation.loc["logistic_ofi", "log_loss"],
            "boosting_val": validation.loc["boosting_ofi", "log_loss"],
            "prior_test": test.loc["baseline", "log_loss"],
            "logistic_test": test.loc["logistic_ofi", "log_loss"],
            "boosting_test": test.loc["boosting_ofi", "log_loss"],
            "top_decile_cents": top["mean_change_dollars"] * 100,
            "bottom_decile_cents": bottom["mean_change_dollars"] * 100,
            "test_trades": int(zero["trades"]),
            "mid_pnl_per_trade_cents": zero["mid_pnl"] / zero["trades"] * 100,
            "spread_cost_per_trade_cents": zero["spread_cost"] / zero["trades"] * 100,
        })
    summary = pd.DataFrame(rows)
    summary["extreme_signal_cents"] = (summary["top_decile_cents"] - summary["bottom_decile_cents"]) / 2
    summary["signal_over_spread"] = summary["extreme_signal_cents"] / summary["spread_cents"]
    summary["cost_over_signal"] = summary["spread_cost_per_trade_cents"] / summary["mid_pnl_per_trade_cents"]
    return summary


def plot_tickers(summary: pd.DataFrame, output_dir: Path) -> None:
    """Relative log loss gain and signal-to-spread ratio across tickers."""

    order = summary.sort_values("spread_cents", ascending=False)
    x = np.arange(len(order))

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(9, 3.4))

    for offset, column, colour, label in (
        (-0.2, "logistic_test", ORANGE, "logistic, 14 features"),
        (0.2, "boosting_test", BLUE, "boosting, 14 features"),
    ):
        gain = 1 - order[column] / order["prior_test"]
        ax_left.bar(x + offset, gain * 100, width=0.38, color=colour, label=label)
    ax_left.set_xticks(x)
    ax_left.set_xticklabels([f"{t}\n{s:.0f}c spread" for t, s in zip(order["ticker"], order["spread_cents"])])
    ax_left.set_ylabel("log loss below the class prior (%)")
    ax_left.set_title("Prediction: test block, one evaluation per name", loc="left", fontsize=9)
    ax_left.legend(frameon=False, fontsize=8)
    ax_left.grid(axis="x", visible=False)

    ax_right.bar(x - 0.2, order["mid_pnl_per_trade_cents"], width=0.38, color=AQUA,
                 label="mid-price move captured per trade")
    ax_right.bar(x + 0.2, order["spread_cost_per_trade_cents"], width=0.38, color=ORANGE,
                 label="spread paid per trade")
    ax_right.set_yscale("log")
    ax_right.set_xticks(x)
    ax_right.set_xticklabels(order["ticker"])
    ax_right.set_ylabel("cents per share (log scale)")
    ax_right.set_title("Execution: aggressive backtest, test block, 0 ms", loc="left", fontsize=9)
    ax_right.legend(frameon=False, fontsize=8, loc="upper right")
    ax_right.set_ylim(top=order["spread_cost_per_trade_cents"].max() * 4)
    ax_right.grid(axis="x", visible=False)
    for i, ratio in enumerate(order["cost_over_signal"]):
        ax_right.annotate(f"×{ratio:.0f}", (i + 0.2, order["spread_cost_per_trade_cents"].iloc[i]),
                          ha="center", va="bottom", fontsize=8, color=TEXT,
                          xytext=(0, 2), textcoords="offset points")

    fig.tight_layout()
    fig.savefig(output_dir / "tickers.png")
    plt.close(fig)


def check_report_inputs(baseline_dir: Path, mlp_dir: Path) -> list[Path]:
    required = [
        baseline_dir / name for name in (
            "validation_metrics.csv", "test_metrics.csv",
            "validation_latency.csv", "test_latency.csv",
            "validation_score_bins.csv", "validation_execution_bins.csv",
        )
    ]
    required += [mlp_dir / "validation_metrics_summary.csv"]
    histories = sorted(mlp_dir.glob("training_history_seed_*.csv"))
    if not histories:
        raise FileNotFoundError(f"No MLP training histories found in {mlp_dir}")
    required += histories
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing report files:\n" + "\n".join(missing))
    for directory in (baseline_dir, mlp_dir):
        status_path = directory / "status.json"
        if status_path.exists():
            status = json.loads(status_path.read_text())
            if status.get("status") != "completed":
                raise ValueError(f"Experiment is not completed: {directory}")
    # New runs must use comparable data, labels, splits and features.
    manifests = [d / "manifest.json" for d in (baseline_dir, mlp_dir)]
    if all(p.exists() for p in manifests):
        a, b = [json.loads(p.read_text()) for p in manifests]
        for key in ("file_prefix", "n_levels", "horizon", "epsilon_units",
                    "train_fraction", "val_fraction", "windows", "depths"):
            if a["configuration"][key] != b["configuration"][key]:
                raise ValueError(f"Incompatible experiments: {key}")
        for key in ("inputs", "ofi_feature_columns"):
            if a[key] != b[key]:
                raise ValueError(f"Incompatible experiments: {key}")
    return required


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path,
                        default=REPORTS_DIR / "baseline_v1")
    parser.add_argument("--mlp-dir", type=Path,
                        default=REPORTS_DIR / "mlp_v1_multiseed")
    parser.add_argument("--uncertainty-dir", type=Path,
                        default=REPORTS_DIR / "uncertainty_v1",
                        help="Decile intervals; skipped when the directory is missing.")
    parser.add_argument("--features-dir", type=Path,
                        default=REPORTS_DIR / "features_v1",
                        help="Feature family results; skipped when missing.")
    parser.add_argument("--passive-dir", type=Path,
                        default=REPORTS_DIR / "passive_v1",
                        help="Passive execution results; skipped when missing.")
    parser.add_argument("--output-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args()
    passive_dir = args.passive_dir if args.passive_dir.is_dir() else None
    uncertainty_dir = args.uncertainty_dir if args.uncertainty_dir.is_dir() else None
    features_dir = args.features_dir if args.features_dir.is_dir() else None
    try:
        inputs = check_report_inputs(args.baseline_dir, args.mlp_dir)
        if uncertainty_dir is not None:
            inputs.append(uncertainty_dir / "validation_decile_intervals.csv")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        plot_model_comparison(args.baseline_dir, args.mlp_dir, args.output_dir)
        plot_score_bins(args.baseline_dir, args.output_dir, uncertainty_dir)
        plot_pnl_decomposition(args.baseline_dir, args.output_dir)
        plot_mlp_training(args.mlp_dir, args.output_dir)
        if features_dir is not None:
            inputs += [
                features_dir / "validation_metrics_by_feature_set.csv",
                features_dir / "permutation_importance.csv",
            ]
            plot_feature_families(features_dir, args.output_dir)
        if passive_dir is not None:
            inputs.append(passive_dir / "validation_passive_summary.csv")
            plot_passive_execution(passive_dir, args.output_dir)
        ticker_summary = summarize_tickers(REPORTS_DIR)
        if len(ticker_summary) > 1:
            for folder in TICKER_DIRS.values():
                for name in ("test_metrics.csv", "test_latency.csv", "validation_score_bins.csv"):
                    path = REPORTS_DIR / folder / name
                    if path.is_file():
                        inputs.append(path)
            ticker_summary.to_csv(args.output_dir / "ticker_summary.csv", index=False)
            plot_tickers(ticker_summary, args.output_dir)
    except (FileNotFoundError, KeyError, ValueError) as error:
        parser.exit(1, f"Cannot build figures: {error}\n")
    provenance = {
        str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in inputs
    }
    (args.output_dir / "figure_sources.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Figures saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
