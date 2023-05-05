# stdlib
from typing import Any, List

# third party
import pandas as pd
from torchvision.transforms import Normalize

# autoprognosis absolute
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.preprocessors.base as base
import autoprognosis.utils.serialization as serialization


class ImageNormalizerPlugin(base.PreprocessorPlugin):
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

    def __init__(self, means: tuple = (0, 0, 0), stds: tuple = (1.0, 1.0, 1.0)) -> None:
        super().__init__()
        self.model = Normalize(mean=means, std=stds)

    @staticmethod
    def name() -> str:
        return "image_normalization"

    @staticmethod
    def subtype() -> str:
        return "preprocessing"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return []

    def _fit(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> "ImageNormalizerPlugin":

        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.model(X)

    def save(self) -> bytes:
        return serialization.save_model(self.model)

    @classmethod
    def load(cls, buff: bytes) -> "ImageNormalizerPlugin":
        model = serialization.load_model(buff)
        return cls(model=model)


plugin = ImageNormalizerPlugin
