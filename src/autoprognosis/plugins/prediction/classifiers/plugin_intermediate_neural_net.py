# stdlib
from typing import Any, List

# third party
import numpy as np
import pandas as pd

# autoprognosis absolute
import autoprognosis.logger as log
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.prediction.classifiers.base as base
from autoprognosis.plugins.prediction.classifiers.plugin_neural_nets import NONLIN
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BasicIntermediateNet(nn.Module):
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
    """

    def __init__(
        self,
        categories_cnt: int,
        n_tab_in: int,
        n_tab_out: int,
        n_img_in: int,
        n_img_out: int,
        n_tab_layer: int = 1,
        n_img_layer: int = 1,
        n_inter_hidden: int = 50,
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
    ) -> None:
        super(BasicIntermediateNet, self).__init__()

        if nonlin not in list(NONLIN.keys()):
            raise ValueError("Unknown nonlinearity")

        NL = NONLIN[nonlin]

        self.n_img_layer = n_img_layer
        self.n_tab_layer = n_tab_layer

        # Tab Net
        if self.n_tab_layer == 3:
            tab_layer = [
                nn.Linear(n_tab_in, self.n_neurons),
                NONLIN[self.non_linear](),
                nn.Linear(n_inter_hidden, n_inter_hidden),
                NONLIN[self.non_linear](),
                nn.Linear(n_inter_hidden, n_tab_out),
            ]
        elif self.n_img_layer == 2:
            tab_layer = [
                nn.Linear(n_tab_in, n_inter_hidden),
                NONLIN[self.non_linear](),
                nn.Linear(n_inter_hidden, n_tab_out),
            ]
        elif self.n_img_layer == 1:
            tab_layer = [nn.Linear(n_tab_in, n_tab_out), NONLIN[self.non_linear]()]
        else:
            tab_layer = [nn.Identity()]

        self.tab_model = nn.Sequential(*tab_layer)

        # Image Net
        if self.n_img_layer == 3:
            img_layer = [
                nn.Linear(n_img_in, n_inter_hidden),
                NONLIN[self.non_linear](),
                nn.Linear(n_inter_hidden, n_inter_hidden),
                NONLIN[self.non_linear](),
                nn.Linear(n_inter_hidden, n_img_out),
            ]
        elif self.n_img_layer == 2:
            img_layer = [
                nn.Linear(n_img_in, self.n_neurons),
                NONLIN[self.non_linear](),
                nn.Linear(self.n_neurons, n_img_out),
            ]
        elif self.n_img_layer == 1:
            img_layer = [nn.Linear(n_tab_in, n_tab_out), NONLIN[self.non_linear]()]
        else:
            img_layer = [nn.Identity()]

        self.image_model = nn.Sequential(*img_layer)

        n_unit_in = n_tab_out + n_img_out

        # Classifier Net
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
        self.classifier_model = nn.Sequential(*layers).to(DEVICE)
        self.categories_cnt = categories_cnt

        self.n_iter = n_iter
        self.batch_size = batch_size
        self.n_iter_print = n_iter_print
        self.patience = patience
        self.n_iter_min = n_iter_min
        self.clipping_value = clipping_value
        self.early_stopping = early_stopping

        self.optimizer = torch.optim.Adam(
            list(self.classifier_model.parameters())
            + list(self.image_model.parameters())
            + list(self.tab_model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )

    def forward(self, X_tab, X_img) -> torch.Tensor:
        x_tab = self.tab_model(X_tab)
        x_img = self.image_model(X_img)
        x_combined = torch.cat((x_tab, x_img), dim=1)
        return self.classifier_model(x_combined)

    def train(self, X: dict, y: torch.Tensor) -> "BasicIntermediateNet":
        X_img = self._check_tensor(X["img"]).float()
        X_tab = self._check_tensor(X["tab"]).float()
        y = self._check_tensor(y).squeeze().long()

        dataset = TensorDataset(X_img, X_tab, y)

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

                X_tab_next, X_img_next, y_next = sample

                preds = self.forward(X_tab_next, X_img_next).squeeze()

                batch_loss = loss(preds, y_next)

                batch_loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    list(self.classifier_model.parameters())
                    + list(self.image_model.parameters())
                    + list(self.tab_model.parameters()),
                    self.clipping_value,
                )

                self.optimizer.step()

                train_loss.append(batch_loss.detach())

            train_loss = torch.Tensor(train_loss).to(DEVICE)

            if self.early_stopping or i % self.n_iter_print == 0:
                with torch.no_grad():
                    X_tab_val, X_img_val, y_val = test_dataset.dataset.tensors

                    preds = self.forward(X_tab_val, X_img_val).squeeze()
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


class IntermediateFusionNeuralNetPlugin(base.ClassifierPlugin):
    """Classification plugin based on a simple intermediate fusion with neural nets for medical images and clinical data

    Method: This plugin uses the data coming from images and tabular data to train a new classifier based on neural nets
    the specificity of this network is that it add a neural nets on top of the tabular and image data before feeding it
    to a new neural nets for classification.


    Issue to solve: the preprocessing can be either by

    Args:
        parameters from neural nets
        n_tabular,
            number of layer on top of tabular data
        n_image,
            number of layer on top of image data
        n_neurons,
            size of the layer
        ratio:
            ratio between image and tabular data (n_image_dim/n_tabular_dim)
    Example:
        >>> from autoprognosis.plugins.prediction import Predictions
        >>> plugin = Predictions(category="classifiers").get("intermediate_fusion_neural_net")
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> plugin.fit_predict(X, y) # returns the probabilities for each class
    """

    def __init__(
        self,
        model: Any = None,
        n_tab_layer: int = 2,
        nonlin="relu",
        n_image_layer: int = 2,
        n_neurons: int = 64,
        ratio: float = 1.0,
        n_layers_hidden: int = 1,
        n_units_hidden: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        dropout: float = 0.1,
        clipping_value: int = 1,
        random_state: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        enable_reproducible_results(random_state)

        self.n_tab_layer = n_tab_layer
        self.n_img_layer = n_image_layer
        self.n_neurons = n_neurons
        self.ratio = ratio
        self.non_linear = nonlin
        self.n_layers_hidden = n_layers_hidden
        self.n_units_hidden = n_units_hidden
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.clipping_value = clipping_value

        #  self.n_iter_print = n_iter_print
        #  self.patience = patience
        #  self.n_iter_min = n_iter_min
        #  self.batch_norm = batch_norm
        #  self.early_stopping = early_stopping

        if model is not None:
            self.model = model
            return

    @staticmethod
    def name() -> str:
        return "intermediate_neural_net"

    @classmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [
            params.Categorical("ratio", [0.5, 1.0, 1.5]),
            params.Integer("n_image_layer", 0, 2),
            params.Integer("n_tab_layer", 0, 2),
            params.Integer("n_tab_layer", 0, 2),
            params.Integer("n_layers_hidden", 1, 2),
            params.Integer("n_units_hidden", 10, 100),
            params.Categorical("lr", [1e-3, 1e-4]),
            params.Categorical("weight_decay", [1e-3, 1e-4]),
            params.Categorical("dropout", [0, 0.1, 0.2]),
            params.Categorical("clipping_value", [0, 1]),
        ]

    def _fit(
        self, X: dict, *args: Any, **kwargs: Any
    ) -> "IntermediateFusionNeuralNetPlugin":

        X_img = torch.from_numpy(np.asarray(X["img"]))
        X_tab = torch.from_numpy(np.asarray(X["tab"]))
        y = args[0]
        y = torch.from_numpy(np.asarray(y))

        self.model = BasicIntermediateNet(
            n_tab_in=X_tab.shape[1],
            n_img_in=X_img.shape[1],
            n_tab_out=50,
            n_img_out=50,
            n_tab_layer=self.n_tab_layer,
            n_img_layer=self.n_img_layer,
            n_inter_hidden=self.n_neurons,
            n_layers_hidden=self.n_layers_hidden,
            n_units_hidden=self.n_units_hidden,
            lr=self.lr,
            weight_decay=self.weight_decay,
            dropout=self.dropout,
            clipping_value=self.clipping_value,
        )

        # Step 2: fit the newly obtained vector with the selected classifier
        self.model.train(X, y)

        return self

    def _predict(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        X_images = kwargs["img"]
        kwargs.pop("img")
        X = pd.DataFrame(np.concatenate((X.to_numpy(), X_images.to_numpy()), axis=1))
        return self.model.predict(X, *args, **kwargs)

    def _predict_proba(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> pd.DataFrame:
        X_images = kwargs["img"]
        kwargs.pop("img")
        X = pd.DataFrame(np.concatenate((X.to_numpy(), X_images.to_numpy()), axis=1))
        return self.model.predict_proba(X, *args, **kwargs)

    def save(self) -> bytes:
        return save_model(self.model)

    @classmethod
    def load(cls, buff: bytes) -> "IntermediateFusionNeuralNetPlugin":
        model = load_model(buff)

        return cls(model=model)


plugin = IntermediateFusionNeuralNetPlugin
