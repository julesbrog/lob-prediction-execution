import numpy as np
import pandas as pd

from lob.splits import make_temporal_splits, prepare_datasets


def test_temporal_splits_purge_labels_at_boundaries():
    n_events = 1_000
    horizon = 50

    splits = make_temporal_splits(
        n_events=n_events,
        horizon=horizon,
        train_fraction=0.6,
        val_fraction=0.2,
    )

    positions = np.arange(n_events)
    train = positions[splits["train"]]
    validation = positions[splits["validation"]]
    test = positions[splits["test"]]

    # An observation at i uses the price at i + horizon as its label.
    assert np.all(train + horizon < validation[0])
    assert np.all(validation + horizon < test[0])
    assert np.all(test + horizon < n_events)

    # The last retained labels reach exactly the preceding block's end.
    assert train[-1] + horizon == validation[0] - 1
    assert validation[-1] + horizon == test[0] - 1
    assert test[-1] + horizon == n_events - 1


def test_prepare_datasets_filters_after_splitting():
    index = pd.RangeIndex(12, name="event_id")

    features = pd.DataFrame(
        {"feature": np.arange(12, dtype=float)},
        index=index,
    )
    targets = pd.DataFrame(
        {
            "target": pd.Series(
                [0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, pd.NA],
                index=index,
                dtype="Int64",
            )
        }
    )

    # Invalid rows must not shift the original split boundaries.
    features.loc[0, "feature"] = np.nan
    features.loc[6, "feature"] = np.inf

    splits = {
        "train": slice(0, 5),
        "validation": slice(5, 9),
        "test": slice(9, 12),
    }

    datasets = prepare_datasets(
        features=features,
        targets=targets,
        splits=splits,
        feature_columns=["feature"],
    )

    expected_indices = {
        "train": [1, 2, 3, 4],
        "validation": [5, 7, 8],
        "test": [9, 10],
    }

    for name, (X, y) in datasets.items():
        assert X.index.tolist() == expected_indices[name]
        assert X.index.equals(y.index)
        assert np.isfinite(X.to_numpy()).all()
        assert y.notna().all()

        pd.testing.assert_series_equal(
            y,
            targets.loc[X.index, "target"].astype("int64"),
        )
