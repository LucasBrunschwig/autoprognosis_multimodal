# stdlib
from typing import Any, List

# third party
import pandas as pd
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor

# autoprognosis absolute
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.preprocessors.base as base
from autoprognosis.utils.default_modalities import IMAGE_KEY
import autoprognosis.utils.serialization as serialization


class ImageResizerPlugin(base.PreprocessorPlugin):
    """Preprocessing plugin for sample normalization based on L2 normalization.

    Method:
        Normalization is the process of scaling individual samples to have unit norm.

    Reference:
        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html

    Example:
        >>> from autoprognosis.plugins.preprocessors import Preprocessors
        >>> plugin = Preprocessors().get("image_resizer")
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> plugin.fit_transform(X, y)
    """

    def __init__(self, size: int = 256) -> None:
        super().__init__()
        self.model = Compose(
            [
                ToTensor(),
                Resize(size=(size, size), antialias=True),
                ToPILImage(),
            ]
        )

    @staticmethod
    def name() -> str:
        return "resizer"

    @staticmethod
    def subtype() -> str:
        return "image_processing"

    @staticmethod
    def modality_type():
        return IMAGE_KEY

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [params.Categorical("size", [32, 124, 256])]

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "ImageResizerPlugin":

        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:

        return pd.DataFrame([self.model(img[0]) for img in X.values])

    def save(self) -> bytes:
        return serialization.save_model(self.model)

    @classmethod
    def load(cls, buff: bytes) -> "ImageResizerPlugin":
        model = serialization.load_model(buff)
        return cls(model=model)


plugin = ImageResizerPlugin
