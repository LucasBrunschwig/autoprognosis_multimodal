# stdlib
from typing import Tuple

# third party
from lifelines.datasets import load_rossi
import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# autoprognosis absolute
from autoprognosis.plugins.explainers.tabular.plugin_lime import plugin
from autoprognosis.plugins.pipeline import Pipeline
from autoprognosis.plugins.prediction.classifiers import Classifiers
from autoprognosis.plugins.prediction.risk_estimation.plugin_cox_ph import (
    plugin as CoxPH,
)
from autoprognosis.plugins.preprocessors import Preprocessors


def dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y = load_breast_cancer(return_X_y=True)
    return train_test_split(X, y, test_size=0.05)


@pytest.mark.slow
@pytest.mark.parametrize("classifier", ["logistic_regression", "xgboost"])
def test_plugin_sanity(classifier: str) -> None:
    X_train, X_test, y_train, y_test = dataset()

    template = Pipeline(
        [
            Preprocessors().get_type("minmax_scaler").fqdn(),
            Classifiers().get_type(classifier).fqdn(),
        ]
    )

    pipeline = template()

    explainer = plugin(pipeline, X_train, y_train, task_type="classification")

    result = explainer.explain(X_test[:2])

    assert len(result) == 2


def test_plugin_name() -> None:
    assert plugin.name() == "lime"


@pytest.mark.slow
def test_plugin_lime_survival_prediction() -> None:
    rossi = load_rossi()

    X = rossi.drop(["week", "arrest"], axis=1)
    Y = rossi["arrest"]
    T = rossi["week"]

    surv = CoxPH().fit(X, T, Y)

    explainer = plugin(
        surv,
        X,
        Y,
        time_to_event=T,
        eval_times=[
            int(T[Y.iloc[:] == 1].quantile(0.50)),
            int(T[Y.iloc[:] == 1].quantile(0.75)),
        ],
        task_type="risk_estimation",
    )

    result = explainer.explain(X.head(1))

    assert result.shape == (1, X.shape[1])
