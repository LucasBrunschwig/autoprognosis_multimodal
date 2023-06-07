# stdlib
from typing import Any, List, Optional

# third party
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from torchvision.transforms import ToTensor

# autoprognosis absolute
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.preprocessors.base as base
from autoprognosis.utils.default_modalities import IMAGE_KEY
import autoprognosis.utils.serialization as serialization


class PCAImagePlugin(base.PreprocessorPlugin):
    """Preprocessing plugin for dimensionality reduction based on the PCA method.

    Method:
        PCA is used to decompose a multivariate dataset in a set of successive orthogonal components that explain a maximum amount of the variance.

    Reference:
        https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

    Args:
        n_components: int
            Number of components to use.

    Example:
        >>> from autoprognosis.plugins.preprocessors import Preprocessors
        >>> plugin = Preprocessors(category="dimensionality_reduction").get("pca_image")
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> plugin.fit_transform(X, y)
    """

    def __init__(
        self, random_state: int = 0, model: Any = None, ratio_pca=2, **kwargs
    ) -> None:
        super().__init__()
        self.random_state = random_state
        self.ratio = ratio_pca
        self.model: Optional[PCA] = None

        if model:
            self.model = model

    @staticmethod
    def name() -> str:
        return "pca_image"

    @staticmethod
    def subtype() -> str:
        return "image_reduction"

    @staticmethod
    def modality_type():
        return IMAGE_KEY

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        # return [params.Categorical("threshold", [0.8, 0.85, 0.9, 0.95])]
        return [params.Categorical("ratio_pca", [1, 2, 3])]

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "PCAImagePlugin":

        if kwargs.get("n_tab", None):
            n_tab = kwargs["n_tab"]
        else:
            n_tab = 60

        X_images = X.squeeze().apply(
            lambda img_: ToTensor()(img_).flatten().detach().cpu().numpy()
        )
        X_images = pd.DataFrame(np.stack(X_images.to_numpy().squeeze()))

        self.model = PCA(
            n_components=int(self.ratio * n_tab), random_state=self.random_state
        )

        self.model.fit(X_images)

        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_images = X.squeeze().apply(
            lambda img_: ToTensor()(img_).flatten().detach().cpu().numpy()
        )
        X_images = pd.DataFrame(np.stack(X_images.to_numpy().squeeze()))
        return self.model.transform(X_images)

    def save(self) -> bytes:
        return serialization.save_model(
            {"model": self.model, "n_components": self.n_components}
        )

    @classmethod
    def load(cls, buff: bytes) -> "PCAImagePlugin":
        args = serialization.load_model(buff)
        return cls(**args)


plugin = PCAImagePlugin
