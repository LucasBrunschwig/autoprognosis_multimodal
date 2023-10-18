# stdlib
from typing import Any, List, Union

# third party
import numpy as np
import pandas as pd

# autoprognosis absolute
from autoprognosis.explorers.core.defaults import CNN as PREDEFINED_CNN, WEIGHTS
from autoprognosis.explorers.core.selector import predefined_args
import autoprognosis.explorers.core.selector as selector
import autoprognosis.plugins.core.params as params
from autoprognosis.plugins.prediction.classifiers.image.plugin_cnn_fine_tune import (
    ConvNetPredefinedFineTune,
    Learning_Rates,
)
import autoprognosis.plugins.preprocessors.base as base
from autoprognosis.plugins.utils.custom_dataset import (
    TestImageDataset,
    build_data_augmentation_strategy,
    data_augmentation_strategies,
)
from autoprognosis.utils.default_modalities import IMAGE_KEY
from autoprognosis.utils.distributions import enable_reproducible_results
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
    conv_name: str,
        the predefined architecture
    n_additional_layer (int):
        the number of added layer to the predefined CNN for transfer learning
    replace_classifier (bool):
        replace the classifier instead of adding layers on top of the classifiers
     non_linear (str):
        the non-linearity in the additional layers
    latent_representation: int
        dimension of latent representation
    lr: int
        learning rate for optimizer. step_size equivalent in the JAX version.
    n_unfrozen_layers:
        the number of layer to unfreeze
    weightec_cross_entropy (bool):
        use weighted cross entropy during training
    weight_decay: float
        l2 (ridge) penalty for the weights.
    n_iter: int
        Maximum number of iterations.
    batch_size: int
        Batch size
    n_iter_print: int
        Number of iterations after which to print updates and check the validation loss.
    patience: int
        Number of iterations to wait before early stopping after decrease in validation loss
    n_iter_min: int
        Minimum number of iterations to go through before starting early stopping
    early_stopping (bool):
        stopping when the metric did not improve for multiple iterations (max = patience)
    clipping_value (int):
        clipping parameters value during training
    random_state: int, default 0
        Random seed

    # Example:
         >>> from autoprognosis.plugins.preprocessors import Preprocessors
         >>> plugin = Preprocessors(category="image_reduction").get("cnn_fine_tune")
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
        # Architecture
        conv_name: str = "AlexNet",
        n_additional_layers: int = 2,
        non_linear: str = "relu",
        replace_classifier: bool = False,
        latent_representation=100,
        # Data Augmentation
        data_augmentation: Union[str, transforms.Compose, None] = "",
        # Training
        lr: int = 3,
        n_unfrozen_layer: int = 2,
        weighted_cross_entropy: bool = False,
        weight_decay: float = 1e-3,
        n_iter: int = 1000,
        batch_size: int = 128,
        n_iter_print: int = 10,
        patience: int = 5,
        n_iter_min: int = 10,
        early_stopping: bool = True,
        clipping_value: int = 1,
        random_state: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        enable_reproducible_results(random_state)

        # Model Architecture
        self.conv_name = conv_name.lower()
        self.non_lin = non_linear
        self.n_additional_layers = n_additional_layers
        self.latent_representation = latent_representation
        self.classifier_removed = False
        self.replace_classifier = replace_classifier
        # Model Fitting
        self.lr = Learning_Rates[lr]
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.n_iter_min = n_iter_min
        self.patience = patience
        self.early_stopping = early_stopping
        self.n_iter_print = n_iter_print
        self.weight_decay = weight_decay
        self.clipping_value = clipping_value
        self.n_unfrozen_layer = n_unfrozen_layer
        self.weighted_cross_entropy = weighted_cross_entropy
        # Data Augmentation Policy
        self.preprocess = self.image_preprocess()
        self.data_augmentation = build_data_augmentation_strategy(data_augmentation)

        # If there are a subgroup of predefined architecture select from it
        if (
            predefined_args.get("predefined_cnn", None)
            and len(predefined_args["predefined_cnn"]) > 0
        ):
            self.conv_name = predefined_args["predefined_cnn"][0]

            # If there are a subgroup of predefined architecture select from it
            if predefined_args.get("latent_representation", None):
                self.latent_representation = predefined_args["latent_representation"][0]

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
        """Hyperparameter Optimization Space"""
        hp_space = [params.Categorical("latent_representation", [50, 100, 300])]

        if not selector.LR_SEARCH:
            hp_space.extend(CNNFeaturesFineTunePlugin.hyperparameter_lr_space())

        return hp_space

    @staticmethod
    def hyperparameter_lr_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        """Hyperparameters to learn optimal latent representation of the images for early fusion"""

        # Subset of predefined architecture
        if (
            predefined_args.get("predefined_cnn", None)
            and len(predefined_args["predefined_cnn"]) > 0
        ):
            CNN = predefined_args["predefined_cnn"]
        else:
            CNN = PREDEFINED_CNN

        return [
            # CNN Architecture
            params.Categorical("conv_name", CNN),
            params.Integer("n_additional_layers", 0, 3),
            params.Categorical("replace_classifier", [True, False]),
            # CNN Fine-Tuning
            params.Integer("lr", 0, 5),
            params.Integer("n_unfrozen_layer", 0, 5),
            params.Categorical("weighted_cross_entropy", [True, False]),
            params.Categorical("clipping_value", [0, 1]),
            # Data Augmentation
            params.Categorical("data_augmentation", data_augmentation_strategies),
        ]

    def sample_hyperparameters(cls, trial, *args: Any, **kwargs: Any):
        param_space = cls.hyperparameter_lr_space(*args, **predefined_args)

        results = {}

        for hp in param_space:
            results[hp.name] = hp.sample(trial)

        return results

    def image_preprocess(self):
        weights = models.get_weight(WEIGHTS[self.conv_name.lower()])
        return weights.transforms(antialias=True)

    def _fit(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> "CNNFeaturesFineTunePlugin":

        if len(*args) == 0:
            raise RuntimeError("Please provide the labels for training")

        y = args[0]

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        self.n_classes = len(y.value_counts())
        y = torch.from_numpy(np.asarray(y))

        self.model = ConvNetPredefinedFineTune(
            conv_name=self.conv_name,
            n_classes=self.n_classes,
            non_linear=self.non_lin,
            transformation=self.data_augmentation,
            batch_size=self.batch_size,
            lr=self.lr,
            n_iter=self.n_iter,
            weight_decay=self.weight_decay,
            early_stopping=self.early_stopping,
            n_iter_print=self.n_iter_print,
            n_iter_min=self.n_iter_min,
            patience=self.patience,
            preprocess=self.preprocess,
            n_unfrozen_layer=self.n_unfrozen_layer,
            n_additional_layers=self.n_additional_layers,
            latent_representation=self.latent_representation,
            weighted_cross_entropy=self.weighted_cross_entropy,
            replace_classifier=self.replace_classifier,
        )

        self.model.train_(X, y)

        return self

    def predict_proba(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        self.model.to(DEVICE)
        self.model.eval()
        with torch.no_grad():
            results = np.empty((0, self.n_classes))
            test_dataset = TestImageDataset(X, preprocess=self.preprocess)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
            for batch_test_ndx, X_test in enumerate(test_loader):
                results = np.vstack(
                    (
                        results,
                        self.model(X_test.to(DEVICE)).detach().cpu().numpy(),
                    )
                )
            self.model.model.train()
            return pd.DataFrame(results)

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.classifier_removed:
            self.model.remove_classification_layer()

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
            self.model.model.train()
            return pd.DataFrame(results)

    def save(self) -> bytes:
        return save_model(self)

    @classmethod
    def load(cls, buff: bytes) -> "CNNFeaturesFineTunePlugin":
        return load_model(buff)


plugin = CNNFeaturesFineTunePlugin
