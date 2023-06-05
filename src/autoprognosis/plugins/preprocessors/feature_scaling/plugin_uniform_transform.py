# stdlib
from typing import Any, List

# third party
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

# autoprognosis absolute
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.preprocessors.base as base
import autoprognosis.utils.serialization as serialization


class UniformTransformPlugin(base.PreprocessorPlugin):
    """Preprocessing plugin for feature scaling based on quantile information.

    Method:
        This method transforms the features to follow a uniform distribution. Therefore, for a given feature, this transformation tends to spread out the most frequent values.

    Reference:
        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html

    Example:
        >>> from autoprognosis.plugins.preprocessors import Preprocessors
        >>> plugin = Preprocessors().get("uniform_transform")
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> plugin.fit_transform(X, y)
    """

    def __init__(
        self, random_state: int = 0, n_quantiles: int = 100, model: Any = None
    ) -> None:
        super().__init__()
        if model:
            self.model = model
            return
        self.model = QuantileTransformer(
            n_quantiles=n_quantiles, random_state=random_state
        )

    @staticmethod
    def name() -> str:
        return "uniform_transform"

    @staticmethod
    def subtype() -> str:
        return "feature_scaling"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return []

    def _fit(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> "UniformTransformPlugin":
        self.model.fit(X)

        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.model.transform(X)

    def save(self) -> bytes:
        return serialization.save_model(self.model)

    @classmethod
    def load(cls, buff: bytes) -> "UniformTransformPlugin":
        model = serialization.load_model(buff)
        return cls(model=model)


plugin = UniformTransformPlugin
