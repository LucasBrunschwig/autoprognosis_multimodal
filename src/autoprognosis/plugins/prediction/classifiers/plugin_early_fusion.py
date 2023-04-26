# stdlib
from typing import Any, List, Optional

# third party
import numpy as np
import pandas as pd
import torchvision

# autoprognosis absolute
import autoprognosis.logger as log
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.prediction.classifiers.base as base
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

WEIGHTS = {
    "alexnet": "AlexNet_Weights.DEFAULT",
    "resnet": "resnet18.DEFAULT",
}

SHAPE = {
    "alexnet": (0, 0, 0),
    "resnet": (3, 250, 250),
}


class ConvNet(nn.Module):
    """Convolutional Neural Network model for early fusion models.

    Parameters
    ----------
    model_name (str):
        model name of one of the default Deep CNN architectures.
    use_pretrained (bool):
        instantiate the model with the latest PyTorch weights of the model.
    fine_tune (bool):
        if false all parameters are set to no grad
    n_features_out (int, optional):
        the number of output features for the ConvNet, if not specified the last layer is replaced by an identity layer.
    preprocess (,optional):
        custom preprocessing for images


    TODO:
    - allow user to use their own weight for a given model.

    """

    def __init__(
        self,
        model_name: str,
        use_pretrained: bool,
        n_features_out: Optional[int],
        fine_tune: bool,
        preprocess: torchvision.transforms.transforms.Compose = None,
    ):

        super(ConvNet, self).__init__()

        self.model_name = model_name
        self.use_pretrained = use_pretrained
        self.n_features_out = n_features_out
        self.fine_tune = fine_tune

        if not self.use_pretrained:
            WEIGHTS[self.model_name] = None

        self.model = models.get_model("alexnet", weights=WEIGHTS[self.model_name]).to(
            DEVICE
        )

        if use_pretrained:
            weights = models.get_weight(WEIGHTS["alexnet"])
            self.preprocess = weights.transforms

        elif preprocess is not None:
            self.preprocess = preprocess

        self.set_parameter_requires_grad()

        if n_features_out is not None:
            # Replace the output layer by the given number of features
            n_features_in = self.model.fc.in_features
            self.model.fc = nn.Linear(n_features_in, n_features_out)
        else:
            # Replace the fc layer by an identity
            self.model.fc = nn.Identity()

    def preprocess_images(self, img_: torch.Tensor) -> torchvision.transforms.Compose:
        return self.preprocess(img_)

    def set_parameter_requires_grad(
        self,
    ):
        if not self.fine_tune:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class EarlyFusionNet(nn.Module):
    """
    Basic neural net.

    Parameters
    ----------
    n_unit_in: int
        Number of features
    categories: int
    n_layers_hidden: int
        Number of hypothesis layers (n_layers_hidden x n_units_hidden + 1 x Linear layer)
    n_units_hidden: int
        Number of hidden units in each hypothesis layer
    nonlin: string, default 'elu'
        Nonlinearity to use in NN. Can be 'elu', 'relu', 'selu' or 'leaky_relu'.
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
    clipping_value: int, default 1
        Gradients clipping value

    TODO: possibility to add any classifier on top if no fine tuning

    """

    def __init__(
        self,
        conv_model_name: str,
        use_pretrained: bool,
        fine_tune: bool,
        n_features_out: int,
        n_tabular: int,
        categories_cnt: int,
        n_layers_hidden: int = 2,
        n_units_hidden: int = 100,
        nonlin: str = "relu",
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        n_iter: int = 1000,
        batch_size: int = 100,
        n_iter_print: int = 10,
        patience: int = 10,
        n_iter_min: int = 100,
        dropout: float = 0.1,
        clipping_value: int = 1,
        batch_norm: bool = False,
        early_stopping: bool = True,
        preprocess: torchvision.transforms.Compose = None,
    ) -> None:
        super(EarlyFusionNet, self).__init__()

        if nonlin not in list(NONLIN.keys()):
            raise ValueError("Unknown nonlinearity")

        # Define the convolutional neural network
        self.conv_model = ConvNet(
            model_name=conv_model_name,
            use_pretrained=use_pretrained,
            n_features_out=n_features_out,
            fine_tune=fine_tune,
            preprocess=preprocess,
        )

        # number of concatenated features
        n_unit_in = (
            self.conv_model(torch.randn(SHAPE[conv_model_name])).shape[1] + n_tabular
        )

        NL = NONLIN[nonlin]

        if n_layers_hidden > 0:
            if batch_norm:
                layers = [
                    nn.Linear(n_unit_in, n_units_hidden),
                    nn.BatchNorm1d(n_units_hidden),
                    NL(),
                ]
            else:
                layers = [nn.Linear(n_unit_in, n_units_hidden), NL()]

            # add required number of layers
            for i in range(n_layers_hidden - 1):
                if batch_norm:
                    layers.extend(
                        [
                            nn.Dropout(dropout),
                            nn.Linear(n_units_hidden, n_units_hidden),
                            nn.BatchNorm1d(n_units_hidden),
                            NL(),
                        ]
                    )
                else:
                    layers.extend(
                        [
                            nn.Dropout(dropout),
                            nn.Linear(n_units_hidden, n_units_hidden),
                            NL(),
                        ]
                    )

            # add final layers
            layers.append(nn.Linear(n_units_hidden, categories_cnt))
        else:
            layers = [nn.Linear(n_unit_in, categories_cnt)]

        layers.append(nn.Softmax(dim=-1))

        # return final architecture
        self.model = nn.Sequential(*layers).to(DEVICE)
        self.categories_cnt = categories_cnt

        self.n_iter = n_iter
        self.batch_size = batch_size
        self.n_iter_print = n_iter_print
        self.patience = patience
        self.n_iter_min = n_iter_min
        self.clipping_value = clipping_value
        self.early_stopping = early_stopping

        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )

    def forward(self, x_img: torch.Tensor, x_tab: torch.Tensor) -> torch.Tensor:
        # Pass the input image through the Inception model
        X_conv = self.conv_model(x_img)

        # Concatenate the output of the CNN with tabular data
        x = torch.cat((X_conv, x_tab), dim=1)

        # Pass the concatenated tensor through the new classifier
        out = self.model(x)

        return out

    def train(self, X: torch.Tensor, y: torch.Tensor) -> "EarlyFusionNet":
        X = self._check_tensor(X).float()
        y = self._check_tensor(y).squeeze().long()

        dataset = TensorDataset(X, y)

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
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

                preds = self.forward(
                    X_next,
                ).squeeze()

                batch_loss = loss(preds, y_next)

                batch_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.parameters(), self.clipping_value)

                self.optimizer.step()

                train_loss.append(batch_loss.detach())

            train_loss = torch.Tensor(train_loss).to(DEVICE)

            if self.early_stopping or i % self.n_iter_print == 0:
                with torch.no_grad():
                    X_val, y_val = test_dataset.dataset.tensors

                    preds = self.forward(
                        X_val,
                    ).squeeze()
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

    def _check_tensor(self, X: torch.Tensor) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            return X.to(DEVICE)
        else:
            return torch.from_numpy(np.asarray(X)).to(DEVICE)


class EarlyFusionPlugin(base.ClassifierPlugin):
    """Classification plugin based early fusion for multimodal data.

    Parameters
    ----------
    n_layers_hidden: int
        Number of hypothesis layers for the classifier module (n_layers_hidden x n_units_hidden + 1 x Linear layer)
    n_units_hidden: int
        Number of hidden units in each hypothesis layer for the classifier module
    nonlin: string, default 'elu'
        Nonlinearity to use in NN. Can be 'elu', 'relu', 'selu' or 'leaky_relu'.
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
    clipping_value: int, default 1
        Gradients clipping value
    random_state: int, default 0
        Random seed


    Example:
        >>> from autoprognosis.plugins.prediction import Predictions
        >>> plugin = Predictions(category="classifiers").get("intermediate_fusion", n_layers_hidden = 2)
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> plugin.fit_predict(X, y) # returns the probabilities for each class
    """

    def __init__(
        self,
        # LUCAS: Convolutional Neural Network
        conv_net: str = "ResNet",
        pre_trained: bool = True,
        # LUCAS: Classifier Parameters
        n_layers_hidden: int = 1,
        n_units_hidden: int = 100,
        nonlin: str = "relu",
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        n_iter: int = 1000,
        batch_size: int = 128,
        n_iter_print: int = 10,
        patience: int = 10,
        n_iter_min: int = 100,
        dropout: float = 0.1,
        clipping_value: int = 1,
        batch_norm: bool = True,
        early_stopping: bool = True,
        hyperparam_search_iterations: Optional[int] = None,
        random_state: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        enable_reproducible_results(random_state)

        if hyperparam_search_iterations:
            n_iter = 5 * int(hyperparam_search_iterations)

        self.n_layers_hidden = n_layers_hidden
        self.n_units_hidden = n_units_hidden
        self.nonlin = nonlin
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.n_iter_print = n_iter_print
        self.patience = patience
        self.n_iter_min = n_iter_min
        self.dropout = dropout
        self.clipping_value = clipping_value
        self.batch_norm = batch_norm
        self.early_stopping = early_stopping

    @staticmethod
    def name() -> str:
        return "early_fusion"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [
            params.Categorical("pretrained_models", ["ResNet", "Inception"]),
            params.Integer("n_layers_hidden", 1, 2),
            params.Integer("n_units_hidden", 10, 100),
            params.Categorical("lr", [1e-3, 1e-4]),
            params.Categorical("weight_decay", [1e-3, 1e-4]),
            params.Categorical("dropout", [0, 0.1, 0.2]),
            params.Categorical("clipping_value", [0, 1]),
        ]

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "EarlyFusionPlugin":
        if len(*args) == 0:
            raise RuntimeError("Please provide the labels for training")

        y = args[0]

        X = torch.from_numpy(np.asarray(X))
        y = torch.from_numpy(np.asarray(y))

        self.model = EarlyFusionNet(
            X.shape[1],
            categories_cnt=len(y.unique()),
            n_layers_hidden=self.n_layers_hidden,
            n_units_hidden=self.n_units_hidden,
            nonlin=self.nonlin,
            lr=self.lr,
            weight_decay=self.weight_decay,
            n_iter=self.n_iter,
            batch_size=self.batch_size,
            n_iter_print=self.n_iter_print,
            patience=self.patience,
            n_iter_min=self.n_iter_min,
            dropout=self.dropout,
            clipping_value=self.clipping_value,
            batch_norm=self.batch_norm,
            early_stopping=self.early_stopping,
        )

        self.model.train(X, y)
        return self

    def _predict(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        with torch.no_grad():
            X = torch.from_numpy(np.asarray(X)).float().to(DEVICE)
            return self.model(X).argmax(dim=-1).detach().cpu().numpy()

    def _predict_proba(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> pd.DataFrame:
        with torch.no_grad():
            X = torch.from_numpy(np.asarray(X)).float().to(DEVICE)
            return self.model(X).detach().cpu().numpy()

    def save(self) -> bytes:
        return save_model(self)

    @classmethod
    def load(cls, buff: bytes) -> "EarlyFusionPlugin":
        return load_model(buff)


plugin = EarlyFusionPlugin
