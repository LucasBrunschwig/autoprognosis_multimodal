# stdlib
from typing import Any, List, Optional

# third party
import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
from torchvision.transforms import ToTensor

# autoprognosis absolute
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.preprocessors.base as base
import autoprognosis.utils.serialization as serialization


class FastICAImagePlugin(base.PreprocessorPlugin):
    """Preprocessing plugin for dimensionality reduction based on Independent Component Analysis algorithm.

    Method:
        Independent component analysis separates a multivariate signal into additive subcomponents that are maximally independent.

    Reference:
        https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html

    Args:
        n_components: int
            Number of components to use.
    Example:
        >>> from autoprognosis.plugins.preprocessors import Preprocessors
        >>> plugin = Preprocessors(category="dimensionality_reduction").get("fast_ica_image")
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> plugin.fit_transform(X, y)
    """

    def __init__(
        self,
        model: Any = None,
        random_state: int = 0,
        ratio_ica: int = 2,
        max_iter=10000,
    ) -> None:
        super().__init__()
        self.random_state = random_state
        self.ratio = ratio_ica
        self.max_iter = max_iter
        self.model: Optional[FastICA] = None

        if model:
            self.model = model

    @staticmethod
    def name() -> str:
        return "fast_ica_image"

    @staticmethod
    def subtype() -> str:
        return "image_reduction"

    @staticmethod
    def modality_type():
        return "image"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [params.Integer("ratio_ica", 1, 4)]

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "FastICAImagePlugin":

        if kwargs.get("n_tab", None):
            self.n_components = self.ratio * kwargs["n_tab"]
        else:
            self.n_components = self.ratio * 60

        X_images = X.squeeze().apply(
            lambda img_: ToTensor()(img_).flatten().detach().cpu().numpy()
        )
        X_images = pd.DataFrame(np.stack(X_images.to_numpy().squeeze()))

        self.model = FastICA(
            n_components=self.n_components,
            random_state=self.random_state,
            max_iter=self.max_iter,
            tol=1e-2,
            whiten="unit-variance",
        )

        self.model.fit(X_images, *args, **kwargs)
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
    def load(cls, buff: bytes) -> "FastICAImagePlugin":
        model = serialization.load_model(buff)
        return cls(**model)


plugin = FastICAImagePlugin
