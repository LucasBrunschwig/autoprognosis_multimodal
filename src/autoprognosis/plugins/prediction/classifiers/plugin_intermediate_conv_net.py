# stdlib
from typing import Any, List

# third party
import numpy as np
import pandas as pd

# autoprognosis absolute
from autoprognosis.explorers.core.defaults import CNN, WEIGHTS
import autoprognosis.logger as log
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.prediction.classifiers.base as base
from autoprognosis.plugins.prediction.classifiers.plugin_neural_nets import NONLIN
from autoprognosis.utils.default_modalities import IMAGE_KEY, TABULAR_KEY
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


class ConvIntermediateNet(nn.Module):
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
        conv_name: str,
        n_img_out: int,
        n_tab_layer: int = 1,
        n_img_layer: int = 1,
        n_inter_hidden: int = 50,
        n_layers_hidden: int = 2,
        n_units_hidden: int = 100,
        n_unfrozen_layer: int = 1,
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
        super(ConvIntermediateNet, self).__init__()

        if nonlin not in list(NONLIN.keys()):
            raise ValueError("Unknown non-linearity")

        NL = NONLIN[nonlin]

        params = []

        # Build the tab layer
        if n_tab_layer > 0:
            if batch_norm:
                tab_layer = [
                    nn.Linear(n_tab_in, n_inter_hidden),
                    nn.BatchNorm1d(n_inter_hidden),
                ]
            else:
                tab_layer = [nn.Linear(n_tab_in, n_inter_hidden)]

            for i in range(n_tab_layer - 1):
                if batch_norm:
                    tab_layer.extend(
                        [
                            nn.Dropout(dropout),
                            nn.Linear(n_inter_hidden, n_inter_hidden),
                            nn.BatchNorm1d(n_inter_hidden),
                            NL(),
                        ]
                    )
                else:
                    tab_layer.extend(
                        [
                            nn.Dropout(dropout),
                            nn.Linear(n_inter_hidden, n_inter_hidden),
                            NL(),
                        ]
                    )
                    # add final layers
            tab_layer.append(nn.Linear(n_inter_hidden, n_tab_out))

        else:
            tab_layer = [nn.Identity()]

        self.tab_model = nn.Sequential(*tab_layer).to(DEVICE)

        for name, param in self.tab_model.named_parameters():
            params.append({"params": param, "lr": lr, "weight_decay": weight_decay})

        # Image Net
        self.image_model = models.get_model(
            conv_name.lower(), weights=WEIGHTS[conv_name.lower()]
        )

        weights = models.get_weight(WEIGHTS[conv_name.lower()])
        self.preprocess = weights.transforms(antialias=True)

        # Unfroze specified layer
        self.set_parameter_requires_grad(n_unfrozen_layer)

        # Replace the output layer by the given number of classes
        if hasattr(self.image_model, "fc"):
            if isinstance(self.image_model.fc, torch.nn.Sequential):
                n_features_in = self.image_model.fc[-1].in_features
            else:
                n_features_in = self.image_model.fc.in_features

        elif hasattr(self.image_model, "classifier"):
            if isinstance(self.image_model.classifier, torch.nn.Sequential):
                n_features_in = self.image_model.classifier[-1].in_features
            else:
                n_features_in = self.image_model.classifier.in_features
        else:
            raise ValueError(f"Unknown last layer type for: {conv_name}")

        # The first intermediate layer depends on the last output
        n_intermediate = n_img_out
        for i in range(n_img_layer):
            n_intermediate *= 2

        additional_layers = [
            nn.Linear(n_features_in, n_intermediate),
            NL(),
            nn.Dropout(p=dropout, inplace=False),
        ]

        for i in range(n_img_layer - 1):
            additional_layers.extend(
                [
                    nn.Linear(n_intermediate, int(n_intermediate / 2)),
                    NL(),
                    nn.Dropout(p=dropout, inplace=False),
                ]
            )
            n_intermediate = int(n_intermediate / 2)
        additional_layers.append(nn.Linear(n_intermediate, n_img_out))

        if hasattr(self.image_model, "fc"):
            self.image_model.fc = nn.Sequential(*additional_layers)
            self.image_model.to(DEVICE)
            for name, param in self.image_model.named_parameters():
                if "fc" in name:
                    params.append(
                        {"params": param, "lr": lr, "weight_decay": weight_decay}
                    )
                elif param.requires_grad:
                    params.append(
                        {"params": param, "lr": 1e-6, "weight_decay": weight_decay}
                    )

        elif hasattr(self.image_model, "classifier"):
            if isinstance(self.image_model.classifier, torch.nn.modules.Sequential):
                self.image_model.classifier[-1] = nn.Sequential(*additional_layers)
                name_match = "classifier." + str(len(self.image_model.classifier) - 1)
            else:
                self.image_model.classifier = nn.Sequential(*additional_layers)
                name_match = "classifier"
            self.image_model.to(DEVICE)
            for name, param in self.image_model.named_parameters():

                if name_match in name:
                    params.append(
                        {"params": param, "lr": lr, "weight_decay": weight_decay}
                    )
                elif param.requires_grad:
                    params.append(
                        {"params": param, "lr": 1e-6, "weight_decay": weight_decay}
                    )

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
        self.classifier_model = nn.Sequential(*layers)
        self.classifier_model.to(DEVICE)

        for name, param in self.classifier_model.named_parameters():
            params.append({"params": param, "lr": lr, "weight_decay": weight_decay})

        self.categories_cnt = categories_cnt

        self.n_iter = n_iter
        self.batch_size = batch_size
        self.n_iter_print = n_iter_print
        self.patience = patience
        self.n_iter_min = n_iter_min
        self.clipping_value = clipping_value
        self.early_stopping = early_stopping

        self.optimizer = torch.optim.Adam(params)

    def forward(self, X_tab, X_img) -> torch.Tensor:
        x_tab = self.tab_model(X_tab)
        x_img = self.image_model(X_img)
        x_combined = torch.cat((x_tab, x_img), dim=1)
        return self.classifier_model(x_combined)

    def train(
        self, X_tab: torch.Tensor, X_img: torch.Tensor, y: torch.Tensor
    ) -> "ConvIntermediateNet":

        X_img = self._check_tensor(X_img).float()
        X_tab = self._check_tensor(X_tab).float()
        y = self._check_tensor(y).squeeze().long()

        dataset = TensorDataset(X_tab, X_img, y)

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size]
        )

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, pin_memory=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, pin_memory=False
        )

        # do training
        val_loss_best = 999999
        patience = 0

        loss = nn.CrossEntropyLoss()

        for i in range(self.n_iter):
            train_loss = []

            for batch_ndx, sample in enumerate(train_loader):
                self.optimizer.zero_grad()

                X_tab_next, X_img_next, y_next = sample
                if torch.cuda.is_available():
                    X_tab_next = X_tab_next.to(DEVICE)
                    X_img_next = X_img_next.to(DEVICE)
                    y_next = y_next.to(DEVICE)

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
                    val_loss = []
                    for batch_test_ndx, val_sample in enumerate(test_loader):
                        X_tab_val, X_img_val, y_val = val_sample

                        if torch.cuda.is_available():
                            X_tab_val = X_tab_val.to(DEVICE)
                            X_img_val = X_img_val.to(DEVICE)
                            y_val = y_val.to(DEVICE)

                        preds = self.forward(X_tab_val, X_img_val).squeeze()
                        val_loss.append(loss(preds, y_val).detach())

                    val_loss = torch.mean(torch.Tensor(val_loss).to(DEVICE))

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
            return X.cpu()
        else:
            return torch.from_numpy(np.asarray(X)).cpu()

    def preprocess_images(self, img_: pd.DataFrame) -> torch.Tensor:
        return torch.stack(img_.squeeze().apply(lambda d: self.preprocess(d)).tolist())

    def set_parameter_requires_grad(
        self,
        num_layers_to_unfreeze: int,
    ):
        for param in self.image_model.parameters():
            param.requires_grad = False

        if num_layers_to_unfreeze > 0:
            unfrozen_layers = 0
            skip = True
            # Iterate over the model modules in reverse order and unfreeze the desired number of layers
            for module in reversed(list(self.image_model.modules())):
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


class IntermediateFusionConvNetPlugin(base.ClassifierPlugin):
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
        n_img_layer: int = 3,
        conv_name: str = "mobilenet_v3_large",
        n_neurons: int = 64,
        ratio: float = 0.96,
        n_layers_hidden: int = 3,
        n_units_hidden: int = 100,
        lr: float = 1e-4,
        weight_decay: float = 1e-3,
        dropout: float = 0.1,
        clipping_value: int = 0,
        random_state: int = 0,
        tab_reduction_ratio=1.0,
        n_iter_print: int = 1,
        patience: int = 20,
        n_iter_min: int = 10,
        n_iter: int = 1000,
        batch_norm: bool = True,
        early_stopping: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        enable_reproducible_results(random_state)

        self.n_tab_layer = n_tab_layer
        self.n_img_layer = n_img_layer
        self.n_neurons = n_neurons
        self.ratio = ratio
        self.non_linear = nonlin
        self.n_layers_hidden = n_layers_hidden
        self.n_units_hidden = n_units_hidden
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.clipping_value = clipping_value
        self.tab_reduction_ratio = tab_reduction_ratio
        self.conv_name = conv_name.lower()

        self.n_iter_print = n_iter_print
        self.patience = patience
        self.n_iter = n_iter
        self.n_iter_min = n_iter_min
        self.batch_norm = batch_norm
        self.early_stopping = early_stopping

        if model is not None:
            self.model = model
            return

    @staticmethod
    def name() -> str:
        return "intermediate_conv_net"

    @staticmethod
    def modality_type() -> str:
        return "multimodal"

    @classmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [
            # Network for Tabular and Image network
            params.Categorical("ratio", [0.6, 0.8, 0.9, 0.95]),
            params.Categorical("tab_reduction_ratio", [0.7, 0.8, 0.9]),
            params.Integer("n_tab_layer", 0, 2),
            params.Integer("n_img_layer", 1, 3),
            params.Categorical("conv_name", CNN),
            # Final Classifiers
            params.Integer("n_layers_hidden", 1, 3),
            params.Integer("n_units_hidden", 10, 100),
            # Training and global parameters
            params.Categorical("lr", [1e-4, 1e-5]),
            params.Categorical("weight_decay", [1e-3, 1e-4]),
            params.Categorical("dropout", [0, 0.1, 0.2]),
            params.Categorical("clipping_value", [0, 1]),
        ]

    def _fit(
        self, X: dict, *args: Any, **kwargs: Any
    ) -> "IntermediateFusionConvNetPlugin":

        X_tab = torch.from_numpy(np.asarray(X[TABULAR_KEY]))
        y = args[0]
        cat = len(np.unique(y))
        y = torch.from_numpy(np.asarray(y))

        n_tab_out = int(self.tab_reduction_ratio * X_tab.shape[1])
        n_img_out = int(n_tab_out / (1 - self.ratio) - n_tab_out)

        self.model = ConvIntermediateNet(
            categories_cnt=cat,
            n_tab_in=X_tab.shape[1],
            conv_name=self.conv_name,
            n_tab_out=n_tab_out,
            n_img_out=n_img_out,
            n_tab_layer=self.n_tab_layer,
            n_img_layer=self.n_img_layer,
            n_inter_hidden=self.n_neurons,
            n_layers_hidden=self.n_layers_hidden,
            n_units_hidden=self.n_units_hidden,
            lr=self.lr,
            weight_decay=self.weight_decay,
            dropout=self.dropout,
            clipping_value=self.clipping_value,
            n_iter_print=self.n_iter_print,
            n_iter_min=self.n_iter_min,
            patience=self.patience,
            n_iter=self.n_iter,
            batch_norm=self.batch_norm,
            early_stopping=self.early_stopping,
        )

        X_img = self.model.preprocess_images(X[IMAGE_KEY])

        # Step 2: fit the newly obtained vector with the selected classifier
        self.model.train(X_tab, X_img, y)

        return self

    def _predict(self, X: dict, *args: Any, **kwargs: Any) -> pd.DataFrame:
        with torch.no_grad():
            X_img = self.model.preprocess_images(X[IMAGE_KEY].squeeze())
            X_img = torch.from_numpy(np.asarray(X_img)).float().to(DEVICE)
            X_tab = torch.from_numpy(np.asarray(X[TABULAR_KEY])).float().to(DEVICE)
            return self.model(X_tab, X_img).argmax(dim=-1).detach().cpu().numpy()

    def _predict_proba(self, X: dict, *args: Any, **kwargs: Any) -> pd.DataFrame:
        with torch.no_grad():
            X_img = self.model.preprocess_images(X[IMAGE_KEY].squeeze())
            X_img = torch.from_numpy(np.asarray(X_img)).float().to(DEVICE)
            X_tab = torch.from_numpy(np.asarray(X[TABULAR_KEY])).float().to(DEVICE)
            return self.model(X_tab, X_img).detach().cpu().numpy()

    def save(self) -> bytes:
        return save_model(self.model)

    @classmethod
    def load(cls, buff: bytes) -> "IntermediateFusionConvNetPlugin":
        model = load_model(buff)

        return cls(model=model)


plugin = IntermediateFusionConvNetPlugin
