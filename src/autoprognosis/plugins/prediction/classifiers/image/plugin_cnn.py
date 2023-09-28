# stdlib
import time
from typing import Any, List, Optional, Union

# third party
import numpy as np
import pandas as pd

# autoprognosis absolute
from autoprognosis.explorers.core.defaults import CNN as PREDEFINED_CNN, CNN_MODEL
from autoprognosis.explorers.core.selector import predefined_args
import autoprognosis.logger as log
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.prediction.classifiers.base as base
from autoprognosis.plugins.utils.custom_dataset import (
    TestImageDataset,
    TrainingImageDataset,
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
        from torch import nn
        from torch.utils.data import DataLoader

        break
    except ImportError:
        depends = ["torch"]
        install(depends)

for retry in range(2):
    try:
        # third party
        from torchvision import transforms

        break
    except ImportError:
        depends = ["torchvision"]
        install(depends)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPS = 1e-8

NONLIN = {
    "elu": nn.ELU,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "selu": nn.SELU,
}

initialization_methods = {
    "": None,
    "kaiming_normal": torch.nn.init.kaiming_normal_,
    "xavier_uniform": torch.nn.init.xavier_uniform_,
    "xavier_normal": torch.nn.init.xavier_normal_,
    "orthogonal": torch.nn.init.orthogonal_,
}
initialization_methods_list = list(initialization_methods.keys())


def initialize_weights(module, init_method_weight="kaiming_normal", model_name="relu"):
    if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
        initialization_methods[init_method_weight](module.weight)

        if module.bias is not None:
            if model_name in ["efficientnet_b4", "mobilenet_v3_large"]:
                module.bias.data.fill(0.0)
            else:  # for ReLU requires non-zero bias
                module.bias.data.fill_(0.01)


class ConvNetPredefined(nn.Module):
    """Convolutional Neural Network model for early fusion models.

    Parameters
    ----------
    model_name (str):
        model name of one of the default Deep CNN architectures.
    use_pretrained (bool):
        instantiate the model with the latest PyTorch weights of the model.
    n_classes (int):
        the number of predicted classes
    n_additional_layer (int):
        the added layer to the predefined CNN for transfer learning
    non_linear (str):
        the non-linearity of the added layers
    batch_size (int):
        batch size for each step during training
    lr (float):
        learning rate for training, usually lower than the initial training
    n_iter (int):
        the number of iteration

    """

    def __init__(
        self,
        model_name: str,
        n_classes: Optional[int],
        non_linear: str,
        transformation: transforms.Compose,
        batch_size: int,
        lr: float,
        n_iter: int,
        weight_decay: float,
        early_stopping: bool,
        n_iter_print: int,
        n_iter_min: int,
        patience: int,
        preprocess,
        n_additional_layers: int = 2,
        clipping_value: int = 1,
        latent_representation: int = None,
        weighted_cross_entropy: bool = False,
        replace_classifier: bool = False,
        init_method: str = "kaiming_normal",
    ):

        super(ConvNetPredefined, self).__init__()

        # Training Parameters
        self.batch_size = batch_size
        self.lr = lr
        self.n_iter = n_iter
        self.n_iter_min = n_iter_min
        self.early_stopping = early_stopping
        self.n_iter_print = n_iter_print
        self.patience = patience
        self.preprocess = preprocess
        self.transforms = transformation
        self.clipping_value = clipping_value
        self.weighted_cross_entropy = weighted_cross_entropy

        # Model Architectures
        self.model_name = model_name.lower()
        self.latent_representation = (
            latent_representation  # latent representation in early fusion
        )
        self.n_additional_layers = n_additional_layers

        # Load predefined CNN without weights
        self.model = CNN_MODEL[self.model_name](weights=None)

        # Initialize weights
        if init_method != "":
            self.model.apply(
                lambda m: initialize_weights(
                    m, init_method_weight=init_method, model_name=self.model_name
                )
            )

        if n_classes is None:
            raise RuntimeError(
                "To build the model, the CNN requires to know the number of classes"
            )

        # Number of features inputs in additional layers
        n_features_in = self.extract_n_features_in(replace_classifier)

        # Define the set of additional layers
        additional_layers = []
        NL = NONLIN[non_linear]

        if n_additional_layers > 0:

            n_intermediate = n_features_in // 2

            additional_layers = [
                nn.Linear(n_features_in, n_intermediate),
                NL(inplace=True),
                nn.Dropout(p=0.5, inplace=False),
            ]

            for i in range(n_additional_layers - 1):
                additional_layers.extend(
                    [
                        nn.Linear(n_intermediate, int(n_intermediate / 2)),
                        NL(inplace=True),
                        nn.Dropout(p=0.5, inplace=False),
                    ]
                )
                n_intermediate = n_intermediate // 2

            # In early fusion, the output size is specified
            if self.latent_representation:
                additional_layers.extend(
                    [
                        nn.Linear(n_intermediate, latent_representation),
                        nn.BatchNorm1d(latent_representation),
                    ]
                )
                additional_layers.append(nn.Linear(latent_representation, n_classes))
                additional_layers[-3].bias.requires_grad = False
            else:
                additional_layers.append(nn.Linear(n_intermediate, n_classes))
        else:
            if self.latent_representation:
                additional_layers.extend(
                    [
                        nn.Linear(n_features_in, latent_representation),
                        nn.BatchNorm1d(latent_representation),
                    ]
                )
                additional_layers.append(nn.Linear(latent_representation, n_classes))
            else:
                additional_layers.append(nn.Linear(n_features_in, n_classes))

        self.integrate_additional_layers(additional_layers, replace_classifier)

        self.model.to(DEVICE)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

    def integrate_additional_layers(self, additional_layers, replace_classifier):
        def integrate_layers(classifier_attr_name):
            """Set up the parameters for optimization"""
            classifier = getattr(self.model, classifier_attr_name)

            if (
                isinstance(classifier, torch.nn.modules.Sequential)
                and not replace_classifier
            ):
                classifier[-1] = nn.Sequential(*additional_layers)
            else:
                setattr(
                    self.model, classifier_attr_name, nn.Sequential(*additional_layers)
                )

        if hasattr(self.model, "fc"):
            integrate_layers("fc")
        elif hasattr(self.model, "classifier"):
            integrate_layers("classifier")

    def extract_n_features_in(self, replace_classifier: bool = False):
        """Extract the size of the input features to define the additional layers.

        Parameters
        ----------
        replace_classifier (bool): specify if the additional layers replace the classifier.
        """

        def get_in_features_from_classifier(module, replace_classifier_=False):
            """Helper function to find the first or last linear input size of the classifier"""
            n_features_in = None
            if isinstance(module, torch.nn.Sequential):
                for layer in module:
                    if isinstance(layer, torch.nn.modules.linear.Linear):
                        n_features_in = layer.in_features
                        if replace_classifier_:
                            break
            elif isinstance(module, torch.nn.modules.linear.Linear):
                n_features_in = module.in_features
            else:
                raise ValueError(f"Unknown Classifier Architecture {self.model_name}")

            return n_features_in

        if hasattr(self.model, "fc"):
            classifier = self.model.fc
        elif hasattr(self.model, "classifier"):
            classifier = self.model.classifier
        else:
            raise ValueError(f"Unknown Classifier Module Name: {self.model_name}")

        return get_in_features_from_classifier(classifier, replace_classifier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_model(self):
        """Returns the model necessary for grad-cam++ computation"""
        return self.model

    def set_zero_grad(self):
        self.model.zero_grad()

    def set_train_mode(self):
        self.model.train()

    def set_eval_mode(self):
        self.model.eval()

    def train(self, X: pd.DataFrame, y: torch.Tensor) -> "ConvNetPredefined":
        y = self._check_tensor(y).squeeze().long()

        dataset = TrainingImageDataset(
            X, y, preprocess=self.preprocess, transform=self.transforms
        )

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_size = len(dataset) - test_size

        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size]
        )

        loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
        )
        val_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
        )

        # do training
        val_loss_best = 999999
        patience = 0
        self.model.train()

        if self.weighted_cross_entropy:
            # TMP LUCAS
            label_counts = torch.bincount(y)
            class_weights = 1.0 / label_counts.float()
            class_weights = class_weights / class_weights.sum()
            class_weights = class_weights.to(DEVICE)
            loss = nn.CrossEntropyLoss(weight=class_weights)
        else:
            loss = nn.CrossEntropyLoss()

        for i in range(self.n_iter):
            train_loss = []
            start_ = time.time()
            for batch_ndx, sample in enumerate(loader):
                self.optimizer.zero_grad()

                X_next, y_next = sample
                X_next = X_next.to(DEVICE)
                y_next = y_next.to(DEVICE)

                preds = self.forward(X_next).squeeze()

                batch_loss = loss(preds, y_next)

                batch_loss.backward()

                self.optimizer.step()

                train_loss.append(batch_loss.detach())

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.clipping_value
                )

            train_loss = torch.Tensor(train_loss).to(DEVICE)

            if self.early_stopping or i % self.n_iter_print == 0:
                with torch.no_grad():

                    val_loss = []

                    for batch_val_ndx, sample in enumerate(val_loader):
                        X_val_next, y_val_next = sample
                        X_val_next = X_val_next.to(DEVICE)
                        y_val_next = y_val_next.to(DEVICE)

                        preds = self.forward(X_val_next).squeeze()
                        val_loss.append(loss(preds, y_val_next).detach())

                    val_loss = torch.mean(torch.Tensor(val_loss))

                    end_ = time.time()

                    if self.early_stopping:
                        if val_loss_best > val_loss:
                            val_loss_best = val_loss
                            patience = 0
                        else:
                            patience += 1

                        if patience > self.patience and i > self.n_iter_min:
                            log.info(
                                f"Final Epoch: {i}, loss: {val_loss:.4f}, train_loss: {torch.mean(train_loss):.4f}"
                            )
                            break

                    if i % self.n_iter_print == 0:
                        log.trace(
                            f"Epoch: {i}, loss: {val_loss:.4f}, train_loss: {torch.mean(train_loss):.4f}, epoch elapsed time: {(end_ - start_):.2f}"
                        )

        return self

    def remove_classification_layer(self):
        if hasattr(self.model, "fc"):
            if isinstance(self.model.fc, nn.Sequential):
                self.model.fc[-1] = nn.Identity()
            else:
                self.model.fc = nn.Identity()
        elif hasattr(self.model, "classifier"):
            if isinstance(self.model.classifier, torch.nn.Sequential):
                if isinstance(self.model.classifier[-1], torch.nn.Sequential):
                    self.model.classifier[-1][-1] = nn.Identity()
                else:
                    self.model.classifier[-1] = nn.Identity()

            elif isinstance(self.model.classifier, torch.nn.Linear):
                self.model.classifier = nn.Identity()
            else:
                raise ValueError(
                    f"Unknown classification layer type - {self.model_name}"
                )

    def _check_tensor(self, X: torch.Tensor) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            return X.cpu()
        else:
            return torch.from_numpy(np.asarray(X)).cpu()


class CNNPlugin(base.ClassifierPlugin):
    """Classification plugin using predefined Convolutional Neural Networks

    Parameters
    ----------
    lr: float
        learning rate for optimizer. step_size equivalent in the JAX version.
    weight_decay: float
        l2 (ridge) penalty for the weights.
    n_iter: int
        Maximum number of iterations.
    batch_size: int
        Batch size
    n_iter_print: int
        Number of iterations after which to print updates and check the validation loss.
    val_split_prop: float
        Proportion of samples used for validation split (can be 0)
    patience: int
        Number of iterations to wait before early stopping after decrease in validation loss
    n_iter_min: int
        Minimum number of iterations to go through before starting early stopping
    random_state: int, default 0
        Random seed


    Example:
         >>> from autoprognosis.plugins.prediction import Predictions
         >>> plugin = Predictions(category="classifier").get("cnn")
         >>> from sklearn.datasets import load_digits
         >>> from PIL import Image
         >>> import numpy as np
         >>> # load data
         >>> X, y = load_digits(return_X_y=True, as_frame=True)
         >>> # Transform X into PIL Images
         >>> X["image"] = X.apply(lambda row: Image.fromarray(np.stack([(row.to_numpy().reshape((8, 8))).astype(np.uint8)]*3, axis=-1)), axis=1)
         >>> plugin.fit_predict(X[["image"]], y)

    #"""

    def __init__(
        self,
        conv_name: str = "alexnet",
        normalisation: bool = "channel-wise",
        non_linear: str = "relu",
        replace_classifier: bool = False,
        size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        n_iter: int = 1000,
        batch_size: int = 100,
        n_iter_print: int = 10,
        data_augmentation: Union[str, transforms.Compose] = None,
        weighted_cross_entropy: bool = False,
        n_additional_layers: int = 2,
        patience: int = 10,
        n_iter_min: int = 10,
        early_stopping: bool = True,
        hyperparam_search_iterations: Optional[int] = None,
        clipping_value: int = 0,
        init_method: str = "",
        random_state: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        enable_reproducible_results(random_state)

        if hyperparam_search_iterations:
            n_iter = 5 * int(hyperparam_search_iterations)

        # CNN Architecture
        self.conv_name = conv_name
        self.replace_classifier = replace_classifier
        self.non_linear = non_linear
        self.n_classes = None  # Defined during training
        self.n_additional_layers = n_additional_layers
        self.init_method = init_method

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
        self.preprocess = None  # defined during training
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
    def modality_type() -> str:
        return IMAGE_KEY

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        if kwargs.get("predefined_cnn", None) and len(kwargs["predefined_cnn"]) > 0:
            CNN = kwargs["predefined_cnn"]
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

    def image_preprocess(self):
        return transforms.Compose(
            [
                transforms.Resize(self.size),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )

    @staticmethod
    def image_to_tensor(img_: pd.DataFrame) -> torch.Tensor:
        img_ = img_.squeeze(axis=1).apply(lambda d: transforms.ToTensor()(d))
        return torch.stack(img_.tolist())

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "CNNPlugin":
        if len(*args) == 0:
            raise RuntimeError("Please provide the labels for training")

        y = args[0]

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        self.compute_normalisation_values(X)
        self.preprocess = self.image_preprocess()

        # Preprocess Data
        self.n_classes = np.unique(y).shape[0]
        y = torch.from_numpy(np.asarray(y))

        self.model = ConvNetPredefined(
            model_name=self.conv_name,
            n_classes=self.n_classes,
            n_additional_layers=self.n_additional_layers,
            lr=self.lr,
            non_linear=self.non_linear,
            n_iter=self.n_iter,
            n_iter_min=self.n_iter_min,
            n_iter_print=self.n_iter_print,
            early_stopping=self.early_stopping,
            patience=self.patience,
            batch_size=self.batch_size,
            weight_decay=self.weight_decay,
            transformation=self.data_augmentation,
            preprocess=self.preprocess,
            clipping_value=self.clipping_value,
            replace_classifier=self.replace_classifier,
            weighted_cross_entropy=self.weighted_cross_entropy,
            init_method=self.init_method,
        )

        self.model.train(X, y)

        return self

    def _predict(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        self.model.to(DEVICE)
        self.model.set_eval_mode()
        with torch.no_grad():
            results = np.empty((0, 1))
            test_loader = DataLoader(
                TestImageDataset(X, preprocess=self.preprocess),
                batch_size=self.batch_size,
                pin_memory=False,
            )
            for batch_test_ndx, X_test in enumerate(test_loader):
                results = np.vstack(
                    (
                        results,
                        np.expand_dims(
                            self.model(X_test.to(DEVICE))
                            .argmax(dim=-1)
                            .detach()
                            .cpu()
                            .numpy(),
                            axis=1,
                        ).astype(int),
                    )
                )
            return pd.DataFrame(results)

    def predict_proba_tensor(self, X: pd.DataFrame):
        """This method forces model to CPU with gradients for grad-CAM++"""
        # TMP LUCAS: check if necessary
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        self.model.cpu()
        self.model.set_eval_mode()
        results = torch.empty((0, self.n_classes))
        test_dataset = TestImageDataset(X, preprocess=self.preprocess)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        for batch_test_ndx, X_test in enumerate(test_loader):
            X_test = X_test.cpu()
            X_test.requires_grad = True
            results = torch.cat(
                (
                    results,
                    self.model(X_test),
                ),
                dim=0,
            )
        return results

    def _predict_proba(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> pd.DataFrame:
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        self.model.to(DEVICE)
        self.model.set_eval_mode()
        with torch.no_grad():
            results = np.empty((0, self.n_classes))
            test_dataset = TestImageDataset(X, preprocess=self.preprocess)
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
            )
            for batch_test_ndx, X_test in enumerate(test_loader):
                results = np.vstack(
                    (
                        results,
                        nn.Softmax(dim=1)(self.model(X_test.to(DEVICE)))
                        .detach()
                        .cpu()
                        .numpy(),
                    )
                )
            return pd.DataFrame(results)

    def set_zero_grad(self):
        self.model.zero_grad()

    def get_image_model(self):
        return self.model.get_model()

    def get_size(self):
        return self.size

    def save(self) -> bytes:
        return save_model(self)

    def get_conv_name(self):
        return self.conv_name

    @classmethod
    def load(cls, buff: bytes) -> "CNNPlugin":
        return load_model(buff)


plugin = CNNPlugin
