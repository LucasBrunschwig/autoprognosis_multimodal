# stdlib
from typing import Any, List

# third party
import pandas as pd

# autoprognosis absolute
from autoprognosis.explorers.core.defaults import CNN, WEIGHTS
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.preprocessors.base as base
from autoprognosis.utils.pip import install
from autoprognosis.utils.serialization import load_model, save_model

for retry in range(2):
    try:
        # third party
        import torch
        from torch import nn

        break
    except ImportError:
        depends = ["torch"]
        install(depends)
for retry in range(2):
    try:
        # third party
        import torchvision.models as models

        break
    except ImportError:
        depends = ["torchvision"]
        install(depends)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPS = 1e-8


class CNNFeaturesPlugin(base.PreprocessorPlugin):
    """Classification plugin using predefined Convolutional Neural Networks

    Parameters
    ----------
    conv_net: str,
        Name of the predefined convolutional neural networks
    random_state: int, default 0
        Random seed


    Example:
        >>> from autoprognosis.plugins.prediction import Predictions
        >>> plugin = Predictions(category="preprocessors").get("predefined_cnn", conv_net='AlexNet')
        >>> from sklearn.datasets import load_iris
        >>> # Load data
        >>> plugin.fit_transform(X, y) # returns the probabilities for each class
    """

    def __init__(
        self,
        conv_net: str = "AlexNet",
        random_state: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.conv_net = conv_net.lower()

        self.model = models.get_model(self.conv_net, weights=WEIGHTS[self.conv_net]).to(
            DEVICE
        )

        weights = models.get_weight(WEIGHTS[self.conv_net])
        self.preprocess = weights.transforms

        if self.conv_net == "alexnet":
            self.model.classifier[6] = nn.Identity()

    @staticmethod
    def name() -> str:
        return "predefined_cnn"

    @staticmethod
    def subtype() -> str:
        return "dimensionality_reduction"

    @staticmethod
    def modality_type():
        return "image"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [params.Categorical("conv_net", CNN)]

    def preprocess_images(self, img_: pd.DataFrame) -> torch.Tensor:
        return torch.stack(img_.apply(lambda d: self.preprocess()(d)).tolist())

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "CNNFeaturesPlugin":
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_preprocess = self.preprocess_images(X.squeeze())
        return pd.DataFrame(self.model(X_preprocess).detach())

    def save(self) -> bytes:
        return save_model(self)

    @classmethod
    def load(cls, buff: bytes) -> "CNNFeaturesPlugin":
        return load_model(buff)


plugin = CNNFeaturesPlugin
