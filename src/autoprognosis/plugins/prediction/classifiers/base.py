# stdlib
from typing import Any

# third party
import pandas as pd

# autoprognosis absolute
import autoprognosis.logger as log
import autoprognosis.plugins.core.base_plugin as plugin
import autoprognosis.plugins.prediction.base as prediction_base
import autoprognosis.plugins.utils.cast as cast
from autoprognosis.utils.default_modalities import IMAGE_KEY
from autoprognosis.utils.tester import classifier_metrics


class ClassifierPlugin(prediction_base.PredictionPlugin):
    """Base class for the classifier plugins.

    It provides the implementation for plugin.Plugin's subtype, _fit and _predict methods.

    Each derived class must implement the following methods(inherited from plugin.Plugin):
        name() - a static method that returns the name of the plugin.
        hyperparameter_space() - a static method that returns the hyperparameters that can be tuned during the optimization. The method will return a list of `Params` derived objects.

    If any method implementation is missing, the class constructor will fail.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.args = kwargs

        super().__init__()

    @staticmethod
    def subtype() -> str:
        return "classifier"

    def fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> plugin.Plugin:

        if isinstance(X, pd.DataFrame):
            X = self._preprocess_training_data(X)
            log.debug(f"Training using {self.fqdn()}, input shape = {X.shape}")

        elif isinstance(X, dict):
            for mod_, df in X.items():
                if mod_ != IMAGE_KEY:
                    X[mod_] = self._preprocess_training_data(df)
                log.debug(
                    f"Training using {self.fqdn()}, {mod_}, input shape = {df.shape}"
                )

        if len(args) == 0:
            raise RuntimeError("Training requires X, y")

        Y = cast.to_dataframe(args[0]).values.ravel()

        self._fit(X, Y, **kwargs)

        self._fitted = True

        if isinstance(X, pd.DataFrame):
            log.debug(f"Done training using {self.fqdn()}, input shape = {X.shape}")

        elif isinstance(X, dict):
            for mod_, df in X.items():
                log.debug(
                    f"Done training using {self.fqdn()}, {mod_}, input shape = {df.shape}"
                )

        return self

    def score(self, X: pd.DataFrame, y: pd.DataFrame, metric: str = "aucroc") -> float:
        ev = classifier_metrics()

        preds = self.predict_proba(X)
        return ev.score_proba(y, preds)[metric]

    def get_args(self) -> dict:
        return self.args
