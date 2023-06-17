# stdlib
from typing import Any, List, Optional

# third party
import numpy as np
import pandas as pd

# autoprognosis absolute
# autoprognosis absolut
from autoprognosis.explorers.core.defaults import CNN, WEIGHTS
import autoprognosis.logger as log
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.prediction.classifiers.base as base
from autoprognosis.utils.default_modalities import IMAGE_KEY
from autoprognosis.utils.distributions import enable_reproducible_results
from autoprognosis.utils.pip import install
from autoprognosis.utils.serialization import load_model, save_model

for retry in range(2):
    try:
        # third party
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

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

NONLIN = {
    "elu": nn.ELU,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "selu": nn.SELU,
}

# Depending on the number of additional layer the intermediate layer have different sizes
N_INTERMEDIATE = {
    "alexnet": {1: 256, 2: 512, 3: 2048},  # last layer is 4096
    "resnet18": {1: 128, 2: 256, 3: 512},  # last layer is 1024
    "resnet50": {1: 128, 2: 256, 3: 512},  # last layer is 1024
    "resnet34": {1: 128, 2: 256, 3: 512},  # last layer is 1024
    "vgg19": {1: 128, 2: 256, 3: 512},  # last layer is 1024
    "mobilenet_v3_large": {1: 128, 2: 256, 3: 512},  # last layer is 1280
    "densenet121": {1: 128, 2: 256, 3: 512},
}


class ConvNetPredefinedFineTune(nn.Module):
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
        the non linearity of the added layers
    batch_size (int):
        batch size for each step during training
    lr (float):
        learning rate for training, usually lower than the initial training
    n_iter (int):
        the number of iteration
    fine_tune (bool):
        define if the weights optimization operates only on the new layer or the whole models

    TODO:
    - allow user to use their own weight for a given model.

    """

    def __init__(
        self,
        model_name: str,
        n_classes: Optional[int],
        non_linear: str,
        batch_size: int,
        lr: float,
        n_iter: int,
        weight_decay: float,
        early_stopping: bool,
        n_iter_print: int,
        n_iter_min: int,
        patience: int,
        n_unfrozen_layer: int = 0,
        n_additional_layer: int = 2,
        n_last_layer: Optional[int] = None,
    ):

        super(ConvNetPredefinedFineTune, self).__init__()

        self.model_name = model_name.lower()

        self.batch_size = batch_size
        self.lr = lr
        self.n_iter = n_iter
        self.n_iter_min = n_iter_min
        self.early_stopping = early_stopping
        self.n_iter_print = n_iter_print
        self.patience = patience

        self.model = models.get_model(
            self.model_name, weights=WEIGHTS[self.model_name]
        ).to(DEVICE)

        weights = models.get_weight(WEIGHTS[self.model_name])
        self.preprocess = weights.transforms

        self.set_parameter_requires_grad(n_unfrozen_layer)

        if n_classes is None:
            raise RuntimeError(
                "To build the architecture, the CNN requires to know the number of classes"
            )

        # Replace the output layer by the given number of classes
        if "resnet" in self.model_name:
            n_features_in = self.model.fc.in_features

        elif self.model_name in [
            "alexnet",
            "vgg19",
            "vgg16",
            "mobilenet_v3_large",
            "densenet121",
        ]:
            if isinstance(self.model.classifier, torch.nn.Sequential):
                n_features_in = self.model.classifier[-1].in_features
            else:
                n_features_in = self.model.classifier.in_features
        NL = NONLIN[non_linear]

        if n_additional_layer > 0:

            if n_last_layer:
                n_intermediate = n_last_layer
                for i in range(n_additional_layer - 1):
                    n_intermediate *= 2
            else:
                n_intermediate = N_INTERMEDIATE[self.model_name][n_additional_layer]
            additional_layers = [
                nn.Linear(n_features_in, n_intermediate),
                NL(),
            ]
            for i in range(n_additional_layer - 1):
                additional_layers.append(
                    nn.Linear(n_intermediate, int(n_intermediate / 2))
                )
                additional_layers.append(NL())
                n_intermediate = int(n_intermediate / 2)
            # Classification layer
            additional_layers.append(nn.Linear(n_intermediate, n_classes))
        else:
            additional_layers = [nn.Linear(n_features_in, n_classes)]

        params = []
        if "resnet" in self.model_name:
            self.model.fc = nn.Sequential(*additional_layers)
            self.model.to(DEVICE)
            for name, param in self.model.named_parameters():
                if "fc" in name:
                    params.append({"params": param, "lr": lr})
                elif param.requires_grad:
                    params.append({"params": param, "lr": 1e-6})
        elif self.model_name in [
            "alexnet",
            "vgg19",
            "vgg16",
            "mobilenet_v3_large",
            "densenet121",
        ]:

            name_match = None
            if isinstance(self.model.classifier, torch.nn.modules.Sequential):
                self.model.classifier[-1] = nn.Sequential(*additional_layers)
                name_match = "classifier." + str(len(self.model.classifier) - 1)
            else:
                self.model.classifier = nn.Sequential(*additional_layers)
                name_match = "classifier"

            self.model.to(DEVICE)
            for name, param in self.model.named_parameters():
                if name_match in name:
                    params.append(
                        {"params": param, "lr": lr, "weight_decay": weight_decay}
                    )
                elif param.requires_grad:
                    params.append(
                        {"params": param, "lr": 1e-6, "weight_decay": weight_decay}
                    )

        self.optimizer = torch.optim.Adam(params)

    def preprocess_images(self, img_: pd.DataFrame) -> torch.Tensor:
        return torch.stack(img_.apply(lambda d: self.preprocess()(d)).tolist())

    def set_parameter_requires_grad(
        self,
        num_layers_to_unfreeze: int,
    ):
        for param in self.model.parameters():
            param.requires_grad = False

        if num_layers_to_unfreeze > 0:
            unfrozen_layers = 0
            skip = True
            # Iterate over the model modules in reverse order and unfreeze the desired number of layers
            for module in reversed(list(self.model.modules())):
                if isinstance(
                    module,
                    (
                        torch.nn.modules.Conv2d,
                        torch.nn.modules.Linear,
                        torch.nn.modules.BatchNorm2d,
                    ),
                ):
                    if skip:
                        skip = False
                        continue
                    for param in module.parameters():
                        param.requires_grad = True
                    unfrozen_layers += 1
                    if unfrozen_layers >= num_layers_to_unfreeze:
                        break
                elif isinstance(module, torch.nn.modules.Sequential):
                    for submodule in reversed(list(module.children())):
                        if isinstance(
                            submodule,
                            (
                                torch.nn.modules.Conv2d,
                                torch.nn.modules.Linear,
                                torch.nn.modules.BatchNorm2d,
                            ),
                        ):
                            if skip:
                                skip = False
                                continue
                            for param in submodule.parameters():
                                param.requires_grad = True
                            unfrozen_layers += 1
                            if unfrozen_layers >= num_layers_to_unfreeze:
                                break
                        elif isinstance(submodule, torch.nn.Sequential):
                            for subsubmodule in reversed(list(submodule.children())):
                                if isinstance(
                                    subsubmodule,
                                    (
                                        torch.nn.modules.Conv2d,
                                        torch.nn.modules.Linear,
                                        torch.nn.modules.BatchNorm2d,
                                    ),
                                ):
                                    if skip:
                                        skip = False
                                        continue
                                    for param in subsubmodule.parameters():
                                        param.requires_grad = True
                                    unfrozen_layers += 1
                                    if unfrozen_layers >= num_layers_to_unfreeze:
                                        break
                if unfrozen_layers >= num_layers_to_unfreeze:
                    break

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def train(self, X: torch.Tensor, y: torch.Tensor) -> "ConvNetPredefinedFineTune":
        X = self._check_tensor(X).float().to(DEVICE)
        y = self._check_tensor(y).squeeze().long().to(DEVICE)

        dataset = TensorDataset(X, y)

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        test_size = min(test_size, 300)
        train_size = len(dataset) - test_size

        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size]
        )

        loader = DataLoader(train_dataset, batch_size=self.batch_size, pin_memory=False)

        # do training
        val_loss_best = 999999
        patience = 0

        loss = nn.CrossEntropyLoss()

        for i in range(self.n_iter):
            train_loss = []

            for batch_ndx, sample in enumerate(loader):
                self.optimizer.zero_grad()

                X_next, y_next = sample

                preds = self.forward(X_next).squeeze()

                batch_loss = loss(preds, y_next)

                batch_loss.backward()

                self.optimizer.step()

                train_loss.append(batch_loss.detach())

            train_loss = torch.Tensor(train_loss).to(DEVICE)

            if self.early_stopping or i % self.n_iter_print == 0:
                with torch.no_grad():
                    X_val, y_val = test_dataset.dataset[test_dataset.indices]

                    preds = self.forward(X_val).squeeze()
                    val_loss = loss(preds, y_val)

                    if self.early_stopping:
                        if val_loss_best > val_loss:
                            val_loss_best = val_loss
                            patience = 0
                        else:
                            patience += 1

                        if patience > self.patience and i > self.n_iter_min:
                            break

                    if i % self.n_iter_print == 0:
                        log.trace(
                            f"Epoch: {i}, loss: {val_loss}, train_loss: {torch.mean(train_loss)}"
                        )

        return self

    def remove_classification_layer(self):
        if self.model_name == "alexnet":
            self.model.classifier[-1] = nn.Identity()
        else:
            self.model.fc = nn.Identity()
            self.model.to(DEVICE)

    def _check_tensor(self, X: torch.Tensor) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            return X.to(DEVICE)
        else:
            return torch.from_numpy(np.asarray(X)).to(DEVICE)


class CNNFineTunePlugin(base.ClassifierPlugin):
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
        >>> plugin = Predictions(category="classifiers").get("cnn", conv_net='ResNet50')
        >>> from sklearn.datasets import load_iris
        >>> # Load data
        >>> plugin.fit_predict(X, y) # returns the probabilities for each class
    """

    def __init__(
        self,
        conv_net: str = "resnet34",
        n_unfrozen_layer: int = 1,
        n_classes: Optional[int] = None,
        n_layer: int = 1,
        non_linear: str = "relu",
        lr: float = 1e-4,
        weight_decay: float = 1e-3,
        n_iter: int = 1000,
        batch_size: int = 64,
        n_iter_print: int = 1,
        patience: int = 5,
        n_iter_min: int = 10,
        early_stopping: bool = True,
        hyperparam_search_iterations: Optional[int] = None,
        random_state: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        enable_reproducible_results(random_state)

        if hyperparam_search_iterations:
            n_iter = 5 * int(hyperparam_search_iterations)

        # Specific to Training
        self.lr = lr
        self.non_linear = non_linear
        self.weight_decay = weight_decay
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.n_iter_print = n_iter_print
        self.patience = patience
        self.n_iter_min = n_iter_min
        self.early_stopping = early_stopping

        # Specific to CNN model
        self.conv_net = conv_net
        self.n_unfrozen_layer = n_unfrozen_layer
        self.n_classes = n_classes
        self.n_additional_layer = n_layer

    @staticmethod
    def name() -> str:
        return "cnn_fine_tune"

    @staticmethod
    def modality_type() -> str:
        return IMAGE_KEY

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [
            params.Categorical("conv_net", CNN),
            params.Categorical("lr", [1e-3, 1e-4, 1e-5]),
            params.Integer("n_layer", 1, 2),
            params.Categorical("non_linear", ["elu", "relu", "leaky_relu", "selu"]),
            params.Integer("n_unfrozen_layer", 0, 3),
        ]

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "CNNFineTunePlugin":
        if len(*args) == 0:
            raise RuntimeError("Please provide the labels for training")

        y = args[0]

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        # Preprocess Data
        n_classes = np.unique(y).shape[0]
        y = torch.from_numpy(np.asarray(y))

        self.model = ConvNetPredefinedFineTune(
            model_name=self.conv_net,
            n_classes=n_classes,
            n_additional_layer=self.n_additional_layer,
            n_unfrozen_layer=self.n_unfrozen_layer,
            lr=self.lr,
            non_linear=self.non_linear,
            n_iter=self.n_iter,
            n_iter_min=self.n_iter_min,
            n_iter_print=self.n_iter_print,
            early_stopping=self.early_stopping,
            patience=self.patience,
            batch_size=self.batch_size,
            weight_decay=self.weight_decay,
        )

        X = self.model.preprocess_images(X.squeeze())

        self.model.train(X, y)

        return self

    def _predict(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        with torch.no_grad():
            X = self.model.preprocess_images(X.squeeze()).to(DEVICE)
            return self.model(X.to(DEVICE)).argmax(dim=-1).detach().cpu().numpy()

    def _predict_proba(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> pd.DataFrame:
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        with torch.no_grad():
            X = self.model.preprocess_images(X.squeeze())
            return self.model(X.to(DEVICE)).detach().cpu().numpy()

    def save(self) -> bytes:
        return save_model(self)

    @classmethod
    def load(cls, buff: bytes) -> "CNNFineTunePlugin":
        return load_model(buff)


plugin = CNNFineTunePlugin
