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
    parser.add_argument("--output-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args()
    uncertainty_dir = args.uncertainty_dir if args.uncertainty_dir.is_dir() else None
    try:
        inputs = check_report_inputs(args.baseline_dir, args.mlp_dir)
        if uncertainty_dir is not None:
            inputs.append(uncertainty_dir / "validation_decile_intervals.csv")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        plot_model_comparison(args.baseline_dir, args.mlp_dir, args.output_dir)
        plot_score_bins(args.baseline_dir, args.output_dir, uncertainty_dir)
        plot_pnl_decomposition(args.baseline_dir, args.output_dir)
        plot_mlp_training(args.mlp_dir, args.output_dir)
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
