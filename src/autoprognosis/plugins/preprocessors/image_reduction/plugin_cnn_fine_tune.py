# stdlib
from typing import Any, List

# third party
import numpy as np
import pandas as pd

# autoprognosis absolute
from autoprognosis.explorers.core.defaults import CNN
import autoprognosis.plugins.core.params as params
from autoprognosis.plugins.prediction.classifiers.plugin_cnn_fine_tune import (
    ConvNetPredefinedFineTune,
)
import autoprognosis.plugins.preprocessors.base as base
from autoprognosis.utils.default_modalities import IMAGE_KEY
from autoprognosis.utils.pip import install
from autoprognosis.utils.serialization import load_model, save_model

for retry in range(2):
    try:
        # third party
        import torch

        break
    except ImportError:
        depends = ["torch"]
        install(depends)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPS = 1e-8


class CNNFeaturesFineTunePlugin(base.PreprocessorPlugin):
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
        nonlin: str = "relu",
        lr: float = 1e-5,
        batch_size: int = 32,
        n_iter: int = 200,
        n_iter_min: int = 10,
        n_iter_print: int = 10,
        patience: int = 5,
        early_stopping: bool = True,
        weight_decay: float = 1e-3,
        n_hidden_units: int = 60,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.conv_net = conv_net.lower()
        self.non_lin = nonlin
        self.lr = lr
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.n_iter_min = n_iter_min
        self.patience = patience
        self.early_stopping = early_stopping
        self.n_iter_print = n_iter_print
        self.weight_decay = weight_decay
        self.n_hidden_units = n_hidden_units

    @staticmethod
    def name() -> str:
        return "predefined_cnn"

    @staticmethod
    def subtype() -> str:
        return "image_reduction"

    @staticmethod
    def modality_type():
        return IMAGE_KEY

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [
            params.Categorical("conv_net", CNN),
            params.Categorical("lr", [1e-4, 1e-5, 1e-6]),
            params.Categorical("n_hidden_units", [30, 60, 120]),
        ]

    def preprocess_images(self, img_: pd.DataFrame) -> torch.Tensor:
        return torch.stack(img_.apply(lambda d: self.preprocess()(d)).tolist())

    def _fit(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> "CNNFeaturesFineTunePlugin":

        y = args[0]

        self.model = ConvNetPredefinedFineTune(
            model_name=self.conv_net,
            n_classes=len(y.value_counts()),
            n_hidden_units=self.n_hidden_units,
            non_linear=self.non_lin,
            weight_decay=self.weight_decay,
            batch_size=self.batch_size,
            lr=self.lr,
            n_iter_min=self.n_iter_min,
            n_iter=self.n_iter,
            early_stopping=self.early_stopping,
            n_iter_print=self.n_iter_print,
            patience=self.patience,
        )

        X_tensor = self.model.preprocess_images(X.squeeze())

        self.model.train(X_tensor, y)

        self.model.remove_classification_layer()

        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        with torch.no_grad():
            X = self.model.preprocess_images(X.squeeze())
            return self.model(X.to(DEVICE)).detach().cpu().numpy()

    def save(self) -> bytes:
        return save_model(self)

    @classmethod
    def load(cls, buff: bytes) -> "CNNFeaturesFineTunePlugin":
        return load_model(buff)


plugin = CNNFeaturesFineTunePlugin
