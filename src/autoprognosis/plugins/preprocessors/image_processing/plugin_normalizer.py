# stdlib
from typing import Any, List

# third party
import numpy as np
import pandas as pd
from torchvision import transforms
from torchvision.transforms import Normalize

# autoprognosis absolute
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.preprocessors.base as base
from autoprognosis.utils.default_modalities import IMAGE_KEY
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

    def __init__(self, apply="channel-wise") -> None:
        super().__init__()
        self.apply = apply
        self.model = None

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
        return [params.Categorical("apply", ["pixel-wise", "channel-wise", "nop"])]

    def _fit(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> "ImageNormalizerPlugin":

        X_images = (
            X.copy()
            .squeeze()
            .apply(lambda img_: transforms.ToTensor()(img_).detach().cpu().numpy())
        )
        X_images = np.stack(X_images.to_numpy().squeeze())

        if self.apply == "channel-wise":
            # Compute mean and stds along each channel
            mean = X_images.mean(axis=(0, 2, 3))
            std = X_images.std(axis=(0, 2, 3))
        elif self.apply == "pixel-wise":
            mean = X_images.mean()
            std = X_images.std()
        elif self.apply == "nop":
            mean = 0
            std = 0
        else:
            raise ValueError("The apply parameters is either pixel- or channel-wise")
        self.model = Normalize(mean=mean, std=std)

        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.apply != "nop":
            return pd.DataFrame(
                X.squeeze().apply(lambda d: self.model(transforms.ToTensor()(d)))
            )
        else:
            return pd.DataFrame(X.squeeze().apply(lambda d: transforms.ToTensor()(d)))

    def save(self) -> bytes:
        return serialization.save_model(self.model)

    @classmethod
    def load(cls, buff: bytes) -> "ImageNormalizerPlugin":
        model = serialization.load_model(buff)
        return cls(model=model)


plugin = ImageNormalizerPlugin
