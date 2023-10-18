# stdlib
from typing import Any, List

# third party
import numpy as np
import pandas as pd

# autoprognosis absolute
from autoprognosis.explorers.core.defaults import (
    CNN as PREDEFINED_CNN,
    CNN_MODEL,
    WEIGHTS,
)
from autoprognosis.explorers.core.selector import predefined_args
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.preprocessors.base as base
from autoprognosis.plugins.utils.custom_dataset import TestImageDataset
from autoprognosis.utils.default_modalities import IMAGE_KEY
from autoprognosis.utils.pip import install
from autoprognosis.utils.serialization import load_model, save_model

for retry in range(2):
    try:
        # third party
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader

        break
    except ImportError:
        depends = ["torch"]
        install(depends)

for retry in range(2):
    try:
        # third party
        from torchvision import models

        break
    except ImportError:
        depends = ["torchvision"]
        install(depends)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPS = 1e-8


class CNNFeaturesImageNetPlugin(base.PreprocessorPlugin):
    """Classification plugin using predefined Convolutional Neural Networks

    Parameters
    ----------
    conv_name: str,
        Name of the predefined convolutional neural networks
    batch_size: int,
        batch size
    random_state: int, default 0
        Random seed


    Example:
         >>> from autoprognosis.plugins.preprocessors import Preprocessors
         >>> plugin = Preprocessors(category="image_reduction").get("cnn_imagenet")
         >>> from sklearn.datasets import load_digits
         >>> from PIL import Image
         >>> import numpy as np
         >>> # load data
         >>> X, y = load_digits(return_X_y=True, as_frame=True)
         >>> # Transform X into PIL Images
         >>> X["image"] = X.apply(lambda row: Image.fromarray(np.stack([(row.to_numpy().reshape((8, 8))).astype(np.uint8)]*3, axis=-1)), axis=1)
         >>> plugin.fit_transform(X[["image"]], y)
    """

    def __init__(
        self,
        conv_name: str = "AlexNet",
        batch_size: int = 128,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.conv_name = conv_name.lower()
        self.batch_size = batch_size

    @staticmethod
    def name() -> str:
        return "cnn_imagenet"

    @staticmethod
    def subtype() -> str:
        return "image_reduction"

    @staticmethod
    def modality_type():
        return IMAGE_KEY

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        if (
            predefined_args.get("predefined_cnn", None)
            and len(predefined_args["predefined_cnn"]) > 0
        ):
            CNN = predefined_args["predefined_cnn"]
        else:
            CNN = PREDEFINED_CNN
        return [params.Categorical("conv_name", CNN)]

    def remove_classification_layer(self):
        if hasattr(self.model, "fc"):
            self.model.fc[-1] = nn.Identity()
        elif hasattr(self.model, "classifier"):
            if isinstance(self.model.classifier, nn.Linear):
                self.model.classifier = nn.Identity()
            else:
                self.model.classifier[-1] = nn.Identity()

    def image_preprocess(self):
        return models.get_weight(WEIGHTS[self.conv_name]).transforms(antialias=True)

    def _fit(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> "CNNFeaturesImageNetPlugin":

        self.model = CNN_MODEL[self.conv_name](weights=WEIGHTS[self.conv_name]).to(
            DEVICE
        )

        self.remove_classification_layer()

        self.preprocess = self.image_preprocess()

        return self

    def preprocess_images(self, img_: pd.DataFrame) -> torch.Tensor:
        return torch.stack(img_.apply(lambda d: self.preprocess()(d)).tolist())

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        self.model.to(DEVICE)
        self.model.eval()
        with torch.no_grad():
            results = np.empty(
                (
                    0,
                    self.model(
                        torch.rand(
                            (
                                3,
                                self.preprocess.resize_size[0],
                                self.preprocess.resize_size[0],
                            )
                        )
                        .unsqueeze(0)
                        .to(DEVICE)
                    ).shape[1],
                )
            )
            test_dataset = TestImageDataset(X, preprocess=self.preprocess)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
            for batch_test_ndx, X_test in enumerate(test_loader):
                results = np.vstack(
                    (
                        results,
                        self.model(X_test.to(DEVICE)).detach().cpu().numpy(),
                    )
                )
            self.model.train()
            return pd.DataFrame(results)

    def save(self) -> bytes:
        return save_model(self)

    @classmethod
    def load(cls, buff: bytes) -> "CNNFeaturesImageNetPlugin":
        return load_model(buff)


plugin = CNNFeaturesImageNetPlugin
