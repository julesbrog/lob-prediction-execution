"""Run a named experiment without overwriting saved results."""

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("baseline_v1", "mlp_multiseed", "uncertainty_v1"))
    parser.add_argument(
        "--output-dir", type=Path,
        help="New output directory (must not exist). Default: reports/runs/<unique ID>.",
    )
    args = parser.parse_args()

    from lob.experiments import ExperimentConfig, run_experiment

    config = ExperimentConfig(project_root=PROJECT_ROOT)
    try:
        run_experiment(args.experiment, config, args.output_dir)
    except FileExistsError:
        parser.error("Output directory already exists. Choose a new --output-dir.")


if __name__ == "__main__":
    main()
