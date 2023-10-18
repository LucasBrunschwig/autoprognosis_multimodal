# stdlib
from typing import Any, List, Union

# third party
import numpy as np
import pandas as pd
from torchvision.transforms import transforms

# autoprognosis absolute
from autoprognosis.explorers.core.defaults import CNN as PREDEFINED_CNN
from autoprognosis.explorers.core.selector import predefined_args
import autoprognosis.explorers.core.selector as selector
import autoprognosis.plugins.core.params as params
from autoprognosis.plugins.prediction.classifiers.image.plugin_cnn import (
    ConvNetPredefined,
    initialization_methods_list,
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


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPS = 1e-8


class CNNFeaturesPlugin(base.PreprocessorPlugin):
    """Image reduction plugin using predefined Convolutional Neural Networks

    Parameters
    ----------
    conv_name (str):,
        the predefined architecture
    init_method (str):,
        change the weights of the network with the selected initialization method
    n_additional_layer (int):
        the number of added layer to the predefined CNN for transfer learning
    non_linear (str):
        non-linearities in additional layers
    replace_classifier (bool):
        replace the classifier instead of adding layers on top of the classifiers
    latent_representation (int):
        dimension of latent representation
    normalisation: str,
        normalisation method name (channel-wise, pixel-wise)
    size (int):,
        size of the resized image during normalisation
    data_augmentation (str):
        data augmentation strategy applied on-the-fly to images during training
    lr: float
        learning rate for optimizer. step_size equivalent in the JAX version.
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


    Example:
        >>> from autoprognosis.plugins.prediction import Predictions
        >>> plugin = Predictions(category="preprocessors").get("predefined_cnn", conv_net='AlexNet')
        >>> from sklearn.datasets import load_iris
        >>> # Load data
        >>> plugin.fit_transform(X, y) # returns the probabilities for each class
    """

    def __init__(
        self,
        # Architecture
        conv_name: str = "alexnet",
        init_method: str = "",
        n_additional_layers: int = 2,
        replace_classifier: bool = False,
        non_linear: str = "relu",
        latent_representation: int = 100,
        # Preprocessing and Data Augmentation
        normalisation: bool = "channel-wise",
        size: int = 256,
        data_augmentation: Union[str, transforms.Compose] = None,
        # Training
        lr: float = 1e-3,
        weighted_cross_entropy: bool = False,
        weight_decay: float = 1e-4,
        n_iter: int = 1000,
        batch_size: int = 100,
        n_iter_print: int = 10,
        patience: int = 10,
        n_iter_min: int = 10,
        early_stopping: bool = True,
        clipping_value: int = 0,
        random_state: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        enable_reproducible_results(random_state)

        # CNN Architecture
        self.conv_name = conv_name
        self.replace_classifier = replace_classifier
        self.non_linear = non_linear
        self.n_classes = None  # Defined during training
        self.n_additional_layers = n_additional_layers
        self.init_method = init_method
        self.latent_representation = latent_representation
        self.classifier_removed = False

        # Training Parameters
        self.lr = lr
        self.weighted_cross_entropy = weighted_cross_entropy
        self.normalisation = normalisation
        self.size = size
        self.weight_decay = weight_decay
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.n_iter_print = n_iter_print
        self.patience = patience
        self.n_iter_min = n_iter_min
        self.early_stopping = early_stopping
        self.clipping_value = clipping_value

        # Data Augmentation
        self.preprocess = None
        self.data_augmentation = build_data_augmentation_strategy(data_augmentation)

        # Ensure the baseline is consistent with selected architecture
        if (
            predefined_args.get("predefined_cnn", None)
            and len(predefined_args["predefined_cnn"]) > 0
        ):
            self.conv_name = predefined_args["predefined_cnn"][0]

    @staticmethod
    def name() -> str:
        return "cnn"

    @staticmethod
    def subtype() -> str:
        return "image_reduction"

    @staticmethod
    def modality_type():
        return IMAGE_KEY

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        """Hyperparameter Optimization Space"""
        hp_space = [params.Categorical("latent_representation", [100])]

        if not selector.LR_SEARCH:
            hp_space.extend(CNNFeaturesPlugin.hyperparameter_lr_space())

        return hp_space

    @staticmethod
    def hyperparameter_lr_space(*args: Any, **kwargs: Any) -> List[params.Params]:
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
            # Training
            params.Integer("lr", 0, 5),
            params.Categorical("init_method", initialization_methods_list),
            params.Categorical("normalisation", ["channel-wise", "pixel-wise"]),
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

    def compute_normalisation_values(self, X: pd.DataFrame):
        local_X = self.image_to_tensor(X.copy())

        if self.normalisation == "channel-wise":
            self.mean = torch.mean(local_X, dim=(0, 2, 3)).tolist()
            self.std = torch.std(local_X, dim=(0, 2, 3)).tolist()
        elif self.normalisation == "pixel-wise":
            self.mean = float(torch.mean(local_X))
            self.std = float(torch.std(local_X))
        else:
            raise ValueError("Unknown normalization type")

    @staticmethod
    def image_to_tensor(img_: pd.DataFrame) -> torch.Tensor:
        img_ = img_.squeeze(axis=1).apply(lambda d: transforms.ToTensor()(d))
        return torch.stack(img_.tolist())

    def image_preprocess(self):
        return transforms.Compose(
            [
                transforms.Resize(self.size),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "CNNFeaturesPlugin":

        if len(*args) == 0:
            raise RuntimeError("Please provide the labels for training")

        y = args[0]
        self.n_classes = len(y.value_counts())
        y = torch.from_numpy(np.asarray(y))

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        self.compute_normalisation_values(X)
        self.preprocess = self.image_preprocess()

        self.model = ConvNetPredefined(
            conv_name=self.conv_name,
            n_classes=self.n_classes,
            non_linear=self.non_linear,
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
            n_additional_layers=self.n_additional_layers,
            clipping_value=self.clipping_value,
            latent_representation=self.latent_representation,
            weighted_cross_entropy=self.weighted_cross_entropy,
            replace_classifier=self.replace_classifier,
            init_method=self.init_method,
        )

        self.model.train(X, y)

        return self

    def predict_proba(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        self.model.to(DEVICE)
        with torch.no_grad():
            self.model.model.eval()
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
        with torch.no_grad():
            self.model.model.eval()
            results = np.empty(
                (
                    0,
                    self.model(
                        torch.rand(
                            (
                                3,
                                self.size,
                                self.size,
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
    def load(cls, buff: bytes) -> "CNNFeaturesPlugin":
        return load_model(buff)


plugin = CNNFeaturesPlugin
