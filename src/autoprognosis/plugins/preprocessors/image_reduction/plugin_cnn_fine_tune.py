# stdlib
from typing import Any, List

# third party
import numpy as np
import pandas as pd

# autoprognosis absolute
from autoprognosis.explorers.core.defaults import CNN as PREDEFINED_CNN, WEIGHTS
from autoprognosis.explorers.core.selector import predefined_args
import autoprognosis.plugins.core.params as params
from autoprognosis.plugins.prediction.classifiers.plugin_cnn_fine_tune import (
    ConvNetPredefinedFineTune,
    TestTensorDataset,
)
import autoprognosis.plugins.preprocessors.base as base
from autoprognosis.utils.default_modalities import IMAGE_KEY
from autoprognosis.utils.pip import install
from autoprognosis.utils.serialization import load_model, save_model

for retry in range(2):
    try:
        # third party
        import torch
        from torch.utils.data import DataLoader

        break
    except ImportError:
        depends = ["torch"]
        install(depends)

for retry in range(2):
    try:
        # third party
        from torchvision import transforms
        import torchvision.models as models

        break
    except ImportError:
        depends = ["torchvision"]
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
        batch_size: int = 128,
        data_augmentation: bool = "rand_augment",
        transformation: transforms.Compose = None,
        n_unfrozen_layer: int = 2,
        n_iter: int = 1000,
        n_iter_min: int = 10,
        n_iter_print: int = 10,
        patience: int = 5,
        early_stopping: bool = True,
        weight_decay: float = 1e-3,
        n_additional_layers: int = 3,
        clipping_value: int = 1,
        output_size=100,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # Model Architecture
        self.conv_net = conv_net.lower()
        self.non_lin = nonlin
        self.n_additional_layers = n_additional_layers
        self.output_size = output_size
        self.classifier_removed = False
        # Model Fitting
        self.lr = lr
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.n_iter_min = n_iter_min
        self.patience = patience
        self.early_stopping = early_stopping
        self.n_iter_print = n_iter_print
        self.weight_decay = weight_decay
        self.clipping_value = clipping_value
        self.n_unfrozen_layer = n_unfrozen_layer
        # Data Augmentation Policy
        self.data_augmentation = data_augmentation
        self.transformation = transformation  # policy to use your own transformation
        self.preprocess = None
        self.transforms_compose = None

        # If there are a subgroup of predefined architecture select from it
        if (
            predefined_args.get("predefined_cnn", None)
            and len(predefined_args["predefined_cnn"]) > 0
        ):
            self.conv_net = predefined_args["predefined_cnn"][0]

        self.image_transform()

    @staticmethod
    def name() -> str:
        return "cnn_fine_tune"

    @staticmethod
    def subtype() -> str:
        return "image_reduction"

    @staticmethod
    def modality_type():
        return IMAGE_KEY

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        # Optimization step: early fusion
        params_output = [params.Categorical("output_size", [50, 100, 300])]
        # Use the learned optimal learning representation features
        search_str = "preprocessor.image_reduction.cnn_fine_tune."
        params_name = [
            "conv_net",
            "n_additional_layers",
            "n_unfrozen_layer",
            "lr",
            "data_augmentation",
        ]
        for name in params_name:
            if kwargs.get(search_str + name, None):
                value = kwargs.get(search_str + name)
                if not isinstance(value, list):
                    value = [value]
                params_output.append(params.Categorical(name, value))

        return params_output

    def hyperparameter_lr_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        # Optimization of learning representation
        if kwargs.get("predefined_cnn", None) and len(kwargs["predefined_cnn"]) > 0:
            CNN = kwargs["predefined_cnn"]
        else:
            CNN = PREDEFINED_CNN

        return [
            # CNN Architecture
            params.Categorical("conv_net", CNN),
            params.Categorical("lr", [1e-4, 1e-5, 1e-6]),
            params.Integer("n_additional_layers", 1, 3),
            # fix the number of unfrozen layers
            params.Integer("n_unfrozen_layer", 0, 4),
            # Use the auto augment policy from pytorch
            params.Categorical(
                "data_augmentation",
                [
                    "",
                    "autoaugment_cifar10",
                    "autoaugment_imagenet",
                    "rand_augment",
                    "trivial_augment",
                ],
            ),
            params.Categorical("clipping_value", [0, 1]),
        ]

    def sample_hyperparameters(cls, trial, *args: Any, **kwargs: Any):
        param_space = cls.hyperparameter_lr_space(*args, **predefined_args)

        results = {}

        for hp in param_space:
            results[hp.name] = hp.sample(trial)

        return results

    def image_transform(self):
        if self.data_augmentation:
            if self.data_augmentation == "autoaugment_imagenet":
                self.transforms = [
                    transforms.AutoAugment(
                        policy=transforms.AutoAugmentPolicy.IMAGENET
                    ),
                ]
            elif self.data_augmentation == "autoaugment_cifar10":
                self.transforms = [
                    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
                ]
            elif self.data_augmentation == "rand_augment":
                self.transforms = [
                    transforms.RandAugment(),
                ]
            elif self.data_augmentation == "trivial_augment":
                self.transforms = [transforms.TrivialAugmentWide()]
        else:
            self.transforms_compose = None

        if self.transformation:
            self.transforms_compose = transforms.Compose(self.transforms)

        weights = models.get_weight(WEIGHTS[self.conv_net.lower()])
        self.preprocess = weights.transforms(antialias=True)

    def _fit(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> "CNNFeaturesFineTunePlugin":

        y = args[0]

        self.n_classes = len(y.value_counts())

        self.model = ConvNetPredefinedFineTune(
            model_name=self.conv_net,
            n_classes=self.n_classes,
            n_additional_layers=self.n_additional_layers,
            n_unfrozen_layer=self.n_unfrozen_layer,
            non_linear=self.non_lin,
            weight_decay=self.weight_decay,
            batch_size=self.batch_size,
            lr=self.lr,
            n_iter_min=self.n_iter_min,
            n_iter=self.n_iter,
            early_stopping=self.early_stopping,
            n_iter_print=self.n_iter_print,
            patience=self.patience,
            preprocess=self.preprocess,
            transformation=self.transforms_compose,
            output_size=self.output_size,
        )

        self.model.train(X, y)

        return self

    def predict_proba(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        self.model.to(DEVICE)
        with torch.no_grad():
            results = np.empty((0, self.n_classes))
            test_dataset = TestTensorDataset(X, preprocess=self.preprocess)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
            for batch_test_ndx, X_test in enumerate(test_loader):
                results = np.vstack(
                    (
                        results,
                        self.model(X_test.to(DEVICE)).detach().cpu().numpy(),
                    )
                )

            return pd.DataFrame(results)

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.classifier_removed:
            self.model.remove_classification_layer()
            self.classifier_removed = True

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        self.model.to(DEVICE)
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
            test_dataset = TestTensorDataset(X, preprocess=self.preprocess)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
            for batch_test_ndx, X_test in enumerate(test_loader):
                results = np.vstack(
                    (
                        results,
                        self.model(X_test.to(DEVICE)).detach().cpu().numpy(),
                    )
                )
            return pd.DataFrame(results)

    def save(self) -> bytes:
        return save_model(self)

    @classmethod
    def load(cls, buff: bytes) -> "CNNFeaturesFineTunePlugin":
        return load_model(buff)


plugin = CNNFeaturesFineTunePlugin
