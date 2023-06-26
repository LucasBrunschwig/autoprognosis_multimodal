# stdlib
from typing import Any, List

# third party
import numpy as np
import pandas as pd

# autoprognosis absolute
from autoprognosis.explorers.core.defaults import MULTIMODAL_KEY
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.preprocessors.base as base
import autoprognosis.utils.serialization as serialization


class ConcatenatePlugin(base.PreprocessorPlugin):
    """Preprocessing plugin for sample normalization based on L2 normalization.

    Method:
        Normalization is the process of scaling individual samples to have unit norm.

    Reference:
        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html

    Example:
        >>> from autoprognosis.plugins.preprocessors import Preprocessors
        >>> plugin = Preprocessors().get("image_normalizer")
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> plugin.fit_transform(X, y)
    """

    def __init__(
        self,
    ) -> None:
        super().__init__()

    @staticmethod
    def name() -> str:
        return "concatenate"

    @staticmethod
    def subtype() -> str:
        return "fusion"

    @staticmethod
    def modality_type():
        return MULTIMODAL_KEY

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return []

    def fit_transform(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Fit the model and transform the training data. Used by imputers and preprocessors."""
        return pd.DataFrame(self.fit(X, *args, *kwargs).transform(X))

    def fit(self, X: dict, *args: Any, **kwargs: Any) -> "ConcatenatePlugin":
        """Train the plugin

        Args:
            X: pd.DataFrame
        """
        self._fit(X, *args, **kwargs)
        self._fitted = True

        return self

    def transform(self, X: dict) -> pd.DataFrame:
        """Transform the input. Used by imputers and preprocessors.

        Args:
            X: pd.DataFrame

        """
        if not self.is_fitted():
            raise RuntimeError("Fit the model first")
        return self.output(self._transform(X))

    def _fit(self, X: dict, *args: Any, **kwargs: Any) -> "ConcatenatePlugin":

        return self

    def _transform(self, X: dict, *args, **kwargs) -> pd.DataFrame:

        if len(X["img"].to_numpy().shape) > 2:
            raise ValueError(
                "Multimodal data points are more than 1-D and can not be concatenated"
            )
        return pd.DataFrame(
            np.concatenate(
                [mod.to_numpy() for mod in X.values() if not mod.empty], axis=1
            )
        )

    def save(self) -> bytes:
        return serialization.save_model(self.model)

    @classmethod
    def load(cls, buff: bytes) -> "ConcatenatePlugin":
        model = serialization.load_model(buff)
        return cls(model=model)


plugin = ConcatenatePlugin
