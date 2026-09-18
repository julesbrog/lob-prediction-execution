import pandas as pd
import numpy as np

from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from copy import deepcopy
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier


def fit_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> DummyClassifier:
    dummy_clf = DummyClassifier(strategy="prior")
    dummy_clf.fit(X_train, y_train)
    return dummy_clf


def fit_logistic(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    solver="lbfgs",
                    C=1.0,
                    max_iter=2000,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    return model


def fit_boosting(
    X_train: pd.DataFrame,
    y_train: pd.Series,
):
    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=15,
        min_samples_leaf=200,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def fit_mlp(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    random_state: int = 42,
    max_epochs: int = 100,
    patience: int = 10,
) -> tuple[Pipeline, pd.DataFrame]:
    """Train an MLP and retain the epoch with lowest validation log loss.

    Temporal splitting and horizon purging must be performed beforehand.
    The scaler is fitted only on training observations.
    """
    for name, value in (
        ("max_epochs", max_epochs),
        ("patience", patience),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise ValueError(f"{name} must be an integer")

        if value <= 0:
            raise ValueError(f"{name} must be strictly positive")

    if isinstance(random_state, (bool, np.bool_)) or not isinstance(
        random_state, (int, np.integer)
    ):
        raise ValueError("random_state must be an integer")

    if not 0 <= random_state < 2**32:
        raise ValueError("random_state must be between 0 and 2**32 - 1")

    classes = np.array([-1, 0, 1])

    for name, X, y in (
        ("train", X_train, y_train),
        ("validation", X_validation, y_validation),
    ):
        if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
            raise ValueError(f"{name}: expected a DataFrame and a Series")

        if X.empty:
            raise ValueError(f"{name}: dataset must not be empty")

        if not X.columns.is_unique:
            raise ValueError(f"{name}: feature names must be unique")

        if not X.index.is_unique or not X.index.equals(y.index):
            raise ValueError(f"{name}: X and y must have matching unique indices")

        values = X.to_numpy(dtype=float, na_value=np.nan)
        if not np.isfinite(values).all():
            raise ValueError(f"{name}: features must be finite")

        if y.isna().any() or not y.isin(classes).all():
            raise ValueError(f"{name}: labels must belong to -1, 0, 1")

    if not X_train.columns.equals(X_validation.columns):
        raise ValueError(
            "train and validation must have identical feature columns in order"
        )

    if not X_train.index.intersection(X_validation.index).empty:
        raise ValueError("train and validation event indices must not overlap")

    if set(y_train.unique()) != set(classes):
        raise ValueError("training data must contain all three classes")

    # Fit preprocessing exclusively on training data.
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(X_train)
    validation_scaled = scaler.transform(X_validation)

    train_labels = y_train.to_numpy(dtype=np.int64)
    validation_labels = y_validation.to_numpy(dtype=np.int64)

    # A persistent generator advances between epochs.
    rng = np.random.RandomState(int(random_state))

    classifier = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        batch_size=min(512, len(X_train)),
        learning_rate_init=1e-3,
        shuffle=True,
        early_stopping=False,
        random_state=rng,
    )

    best_classifier = None
    best_loss = np.inf
    best_epoch = 0
    epochs_without_improvement = 0
    records = []

    for epoch in range(1, int(max_epochs) + 1):
        # One training epoch, preserving weights and optimizer state.
        if epoch == 1:
            classifier.partial_fit(
                train_scaled,
                train_labels,
                classes=classes,
            )
        else:
            classifier.partial_fit(train_scaled, train_labels)

        train_probabilities = classifier.predict_proba(train_scaled)
        validation_probabilities = classifier.predict_proba(validation_scaled)

        train_loss = log_loss(
            train_labels,
            train_probabilities,
            labels=classifier.classes_,
        )
        validation_loss = log_loss(
            validation_labels,
            validation_probabilities,
            labels=classifier.classes_,
        )

        if not np.isfinite(train_loss) or not np.isfinite(validation_loss):
            raise RuntimeError("MLP training produced a non-finite log loss")

        records.append(
            {
                "epoch": epoch,
                "train_log_loss": float(train_loss),
                "validation_log_loss": float(validation_loss),
            }
        )

        # Copy the complete estimator, not a reference to the live model.
        if validation_loss < best_loss:
            best_loss = float(validation_loss)
            best_epoch = epoch
            best_classifier = deepcopy(classifier)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:03d} | "
            f"train log loss: {train_loss:.6f} | "
            f"validation log loss: {validation_loss:.6f}",
            flush=True,
        )

        if epochs_without_improvement >= patience:
            print(
                f"Early stopping: {patience} epochs without improvement.",
                flush=True,
            )
            break

    history = pd.DataFrame(records)
    history["is_best"] = history["epoch"].eq(best_epoch)

    # Both steps are already fitted: do not call fit() on this pipeline.
    model = Pipeline(
        [
            ("scaler", scaler),
            ("classifier", best_classifier),
        ]
    )

    print(
        f"Restored epoch {best_epoch} (validation log loss: {best_loss:.6f}).",
        flush=True,
    )

    return model, history
