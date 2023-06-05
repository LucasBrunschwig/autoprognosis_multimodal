# stdlib
from typing import Any, List

# third party
import numpy as np
import pandas as pd
from torchvision import transforms
from torchvision.transforms import Normalize

# autoprognosis absolute
from autoprognosis.explorers.core.defaults import IMAGE_KEY
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

    def __init__(
        self, apply=True, means: tuple = (0, 0, 0), stds: tuple = (1.0, 1.0, 1.0)
    ) -> None:
        super().__init__()
        self.model = Normalize(mean=means, std=stds)
        self.apply = apply

    @staticmethod
    def name() -> str:
        return "normalizer"

    @staticmethod
    def subtype() -> str:
        return "image_processing"

    @staticmethod
    def modality_type():
        return IMAGE_KEY

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [params.Categorical("apply", [True, False])]

    def _fit(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> "ImageNormalizerPlugin":

        X_images = X.squeeze().apply(
            lambda img_: transforms.ToTensor()(img_).detach().cpu().numpy()
        )
        X_images = np.stack(X_images.to_numpy().squeeze())

        # Compute mean and stds along each channel
        mean = X_images.mean(axis=(0, 2, 3))
        std = X_images.std(axis=(0, 2, 3))

        self.model = Normalize(mean=mean, std=std)

        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.apply:
            return pd.DataFrame(
                X.squeeze().apply(
                    lambda d: transforms.ToPILImage()(
                        self.model(transforms.ToTensor()(d))
                    )
                )
            )
        else:
            return X

    def save(self) -> bytes:
        return serialization.save_model(self.model)

    @classmethod
    def load(cls, buff: bytes) -> "ImageNormalizerPlugin":
        model = serialization.load_model(buff)
        return cls(model=model)


plugin = ImageNormalizerPlugin
