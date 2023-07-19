# stdlib
from typing import Any, List

# third party
import numpy as np
import pandas as pd

# autoprognosis absolute
from autoprognosis.explorers.core.defaults import MULTIMODAL_KEY
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.preprocessors.base as base
from autoprognosis.utils.default_modalities import IMAGE_KEY
import autoprognosis.utils.serialization as serialization


class ConcatenatePlugin(base.PreprocessorPlugin):
    """Preprocessing plugin for modality fusion

    Method:
       simply concatenate the data into a unique vector

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

    def _fit(self, X: dict, *args: Any, **kwargs: Any) -> "ConcatenatePlugin":
        return self

    def _transform(self, X: dict, *args, **kwargs) -> pd.DataFrame:

        if len(X[IMAGE_KEY].to_numpy().shape) > 2:
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
