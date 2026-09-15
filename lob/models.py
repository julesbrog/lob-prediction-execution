import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


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
