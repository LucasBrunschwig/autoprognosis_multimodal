# stdlib
import time
from typing import Any, List

# third party
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

# autoprognosis absolute
from autoprognosis.explorers.core.defaults import CNN as PREDEFINED_CNN, WEIGHTS
from autoprognosis.explorers.core.selector import predefined_args
import autoprognosis.logger as log
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.prediction.classifiers.base as base
from autoprognosis.plugins.utils.custom_dataset import (
    TestMultimodalDataset,
    TrainingMultimodalDataset,
    build_data_augmentation_strategy,
    data_augmentation_strategies,
)
from autoprognosis.utils.default_modalities import IMAGE_KEY, TABULAR_KEY
from autoprognosis.utils.distributions import enable_reproducible_results
from autoprognosis.utils.pip import install
from autoprognosis.utils.serialization import load_model, save_model

for retry in range(2):
    try:
        # third party
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Subset

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


class MetaBlock(nn.Module):
    """
    Metadata Processing Block (MetaBlock), gated mechanism for each feature maps
    """

    def __init__(self, V, U):
        super(MetaBlock, self).__init__()
        self.fb = nn.Sequential(nn.Linear(U, V), nn.BatchNorm1d(V))
        self.gb = nn.Sequential(nn.Linear(U, V), nn.BatchNorm1d(V))

    def forward(self, V, U):
        t1 = self.fb(U)
        t2 = self.gb(U)
        V = torch.sigmoid(torch.tanh(V * t1.unsqueeze(-1)) + t2.unsqueeze(-1))
        return V


class MetaBlockArchitecture(nn.Module):
    """MetabBlock Model largely inspired from Pacheco and Kroling 2021.

    Parameters:
    ----------
    conv_name: str,
        the name of the predefined CNN
    n_classes: int,
        the final number of classes
    n_metadata: int,
        the number of tabular features
    transform:
        data augmentation transformations
    preprocess:
        image preprocessing
    n_reducer_neurons: int
        number of hidden neurons in classifier
    n_reducer_layer: int
        number of hidden layers in classifier
    freeze_conv: bool,
        freeze the convolutional layer of the network
    dropout: float,
        .
    n_iter_print: int
        .
    n_iter_min: int
        .
    patience: int
        .
    n_iter: int
        .
    lr: float,
        .
    weight_decay: float
        .
    clipping_value: int
        .
    batch_norm: bool,
        .
    early_stopping: bool
        .
    batch_size: int,
        .

    """

    def __init__(
        self,
        n_classes: int,
        n_metadata: int,
        conv_name: str,
        transform: transforms.Compose,
        preprocess,
        n_reducer_neurons: int = 256,
        n_reducer_layer: int = 1,
        freeze_conv=False,
        dropout: float = 0.5,
        weighted_cross_entropy: bool = False,
        n_iter_print: int = 10,
        n_iter_min: int = 10,
        patience: int = 10,
        n_iter: int = 1000,
        lr: float = 1e-4,
        weight_decay: float = 1e-3,
        clipping_value: int = 1,
        batch_norm: bool = False,
        early_stopping: bool = True,
        batch_size: int = 100,
    ):

        super(MetaBlockArchitecture, self).__init__()

        # Architecture
        self.conv_name = conv_name
        conv_model = models.get_model(
            conv_name.lower(), weights=WEIGHTS[conv_name.lower()]
        )

        self.features = nn.Sequential(*list(conv_model.children())[:-1])
        self.n_reducer_layer = n_reducer_layer
        self.n_reducer_neurons = n_reducer_neurons
        self.n_reducer_layer = n_reducer_layer
        self.dropout = dropout

        # Training
        self.batch_norm = batch_norm
        self.preprocess = preprocess
        self.transform = transform
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.clipping_value = clipping_value
        self.patience = patience
        self.early_stopping = early_stopping
        self.weighted_cross_entropy = weighted_cross_entropy

        # Miscellaneous
        self.n_iter = n_iter
        self.n_iter_min = n_iter_min
        self.n_iter_print = n_iter_print

        self.n_feat_map, self.n_feat_map_size = self.feature_maps_size()
        self.feat_size = self.n_feat_map_size**2 * self.n_feat_map

        self.combination = MetaBlock(self.n_feat_map, n_metadata)

        # Freeze CNN layers
        if freeze_conv:
            for param in self.features.parameters():
                param.requires_grad = False

        # Feature reducer base
        if n_reducer_layer > 0:
            additional_layers = [
                nn.Linear(self.feat_size, n_reducer_neurons),
                nn.BatchNorm1d(n_reducer_neurons),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            ]
            for i in range(n_reducer_layer - 1):
                additional_layers.append(
                    nn.Linear(n_reducer_neurons, n_reducer_neurons // 2)
                )
                additional_layers.append(nn.BatchNorm1d(n_reducer_neurons // 2))
                additional_layers.append(nn.ReLU())
                additional_layers.append(nn.Dropout(p=dropout))
                n_reducer_neurons = n_reducer_neurons // 2

            self.reducer_block = nn.Sequential(*additional_layers)
            self.classifier = nn.Linear(n_reducer_neurons, n_classes)

        else:
            self.reducer_block = None
            self.classifier = nn.Linear(self.feat_size, n_classes)

        self.to(DEVICE)

        if self.reducer_block:
            self.optimizer = torch.optim.Adam(
                list(self.features.parameters())
                + list(self.combination.parameters())
                + list(self.reducer_block.parameters())
                + list(self.classifier.parameters()),
                lr=lr,
                weight_decay=weight_decay,
            )
        else:
            self.optimizer = torch.optim.Adam(
                list(self.features.parameters())
                + list(self.combination.parameters())
                + list(self.classifier.parameters()),
                lr=lr,
                weight_decay=weight_decay,
            )

    def feature_maps_size(self):
        dummy_input = torch.randn(1, 3, 224, 224)  # random input
        output = self.features(dummy_input).detach().cpu()
        return output.shape[1], output.shape[2]  # the size of feature maps

    def forward(self, meta_data, img):
        x = self.features(img)
        x = x.view(x.size(0), self.n_feat_map, self.n_feat_map_size**2, -1).squeeze(
            -1
        )  # reshaping the feature maps
        x = self.combination.forward(x, meta_data.float())  # applying MetaBlock
        x = x.view(x.size(0), -1)
        if self.reducer_block:
            x = self.reducer_block(x)  # feature reducer block
        return self.classifier(x)

    def train(
        self, X_tab: torch.Tensor, X_img: pd.DataFrame, y: torch.Tensor
    ) -> "MetaBlockArchitecture":

        X_tab = self._check_tensor(X_tab).float()
        y = self._check_tensor(y).squeeze().long()

        dataset = TrainingMultimodalDataset(
            X_tab, X_img, y, weight_transform=self.preprocess, transform=self.transform
        )

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_indices, test_indices = next(sss.split(X_tab, y))

        train_dataset = Subset(dataset, train_indices)
        test_dataset = Subset(dataset, test_indices)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, pin_memory=True, drop_last=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, pin_memory=True, drop_last=True
        )

        # do training
        val_loss_best = 999999
        patience = 0

        self.train_()

        if self.weighted_cross_entropy:
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

                if self.reducer_block:
                    torch.nn.utils.clip_grad_norm_(
                        (
                            list(self.classifier.parameters())
                            + list(self.features.parameters())
                            + list(self.combination.parameters())
                            + list(self.reducer_block.parameters())
                        ),
                        self.clipping_value,
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(
                        (
                            list(self.classifier.parameters())
                            + list(self.features.parameters())
                            + list(self.combination.parameters())
                        ),
                        self.clipping_value,
                    )

                self.optimizer.step()

                train_loss.append(batch_loss.detach())

            train_loss = torch.Tensor(train_loss).to(DEVICE)
            end_ = time.time()

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
                            log.trace(
                                f"Final Epoch: {i}, loss: {val_loss:.4f}, train_loss: {torch.mean(train_loss):.4f}, "
                                f"elapsed time {(end_ - start_):.2f}"
                            )
                            break

                    if i % self.n_iter_print == 0:
                        log.trace(
                            f"Epoch: {i}, loss: {val_loss:.4f}, train_loss: {torch.mean(train_loss):.4f}, "
                            f"elapsed time {(end_ - start_):.2f}"
                        )

        return self

    def _check_tensor(self, X: torch.Tensor) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            return X.cpu()
        else:
            return torch.from_numpy(np.asarray(X)).cpu()

    def zero_grad_model(self):
        self.features.zero_grad()
        if self.reducer_block:
            self.reducer_block.zero_grad()
        self.combination.zero_grad()
        self.classifier.zero_grad()

    def train_(self):
        self.features.train()
        if self.reducer_block:
            self.reducer_block.train()
        self.combination.train()
        self.classifier.train()

    def get_image_model(self):
        return self.features


class MetaBlockPlugin(base.ClassifierPlugin):
    """Classification plugin based on a simple intermediate fusion with neural nets for medical images and clinical data

    Method: This plugin uses the data coming from images and tabular data to train a new classifier based on neural nets
    the specificity of this network is that it add a neural nets on top of the tabular and image data before feeding it
    to a new neural nets for classification.


    Issue to solve: the preprocessing can be either by

    Args:
        Parameters:
        ----------
        model: Any,
            existing model
        n_reducer_neurons: int
            number of hidden neurons in classifier
        n_reducer_layer: int
            number of hidden layers in classifier
        conv_name: str,
            the name of the predefined CNN
        data_augmenetation: str
            data augmentation strategy
        dropout: float,
            .
        n_unfrozen_layers: int
            number of unfrozen layers
        lr: float,
            .
        weight_decay: float
            .
        clipping_value: int
            .
        batch_norm: bool
            .
        weighted_cross_entropy: bool,
            .
        early_stopping: bool
            .
        batch_size: int,
            .
        patience: int
            .
        n_iter: int
            .
        n_iter_min: int
            .
        n_iter_print: int
            .
        random_state: int,
            .

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
        # Network Architecture
        n_reducer_layer: int = 1,
        n_reducer_neurons: int = 1024,
        conv_name: str = "alexnet",
        # Training
        data_augmentation: str = "",
        dropout: float = 0.5,
        n_unfrozen_layers: int = 3,
        lr: float = 1e-4,
        weight_decay: float = 1e-3,
        clipping_value: int = 1,
        batch_norm: bool = True,
        weighted_cross_entropy: bool = False,
        early_stopping: bool = True,
        batch_size: int = 64,
        patience: int = 5,
        n_iter: int = 1000,
        n_iter_min: int = 10,
        # Miscellaneous
        n_iter_print: int = 1,
        random_state: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        enable_reproducible_results(random_state)

        # Architecture
        self.n_reducer_neurons = n_reducer_neurons
        self.n_reducer_layer = n_reducer_layer
        self.conv_name = conv_name.lower()

        # Training
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.clipping_value = clipping_value
        self.conv_name = conv_name.lower()
        self.batch_size = batch_size
        self.batch_norm = batch_norm
        self.early_stopping = early_stopping
        self.n_unfrozen_layers = n_unfrozen_layers
        self.weighted_cross_entropy = weighted_cross_entropy

        # Miscellaneous
        self.n_iter_print = n_iter_print
        self.patience = patience
        self.n_iter = n_iter
        self.n_iter_min = n_iter_min

        self.preprocess = self.image_preprocessing()
        self.data_augmentation = build_data_augmentation_strategy(data_augmentation)

        # Ensure baseline is consistent with selected architecture
        if (
            predefined_args.get("predefined_cnn", None)
            and len(predefined_args["predefined_cnn"]) > 0
        ):
            self.conv_name = predefined_args["predefined_cnn"][0]

        if model is not None:
            self.model = model
            return

    def image_preprocessing(self):
        weights = models.get_weight(WEIGHTS[self.conv_name.lower()])
        return weights.transforms(antialias=True)

    @staticmethod
    def name() -> str:
        return "metablock"

    @staticmethod
    def modality_type() -> str:
        return "multimodal"

    @classmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        if kwargs.get("predefined_cnn", None) and len(kwargs["predefined_cnn"]) > 0:
            CNN = kwargs["predefined_cnn"]
        else:
            CNN = PREDEFINED_CNN

        return [
            # Network for Tabular and Image network
            params.Integer("n_reducer_layer", 0, 3),
            params.Categorical("n_reducer_neurons", [256, 512, 1024, 2048]),
            params.Categorical("conv_name", CNN),
            # Training and global parameters
            params.Categorical("lr", [1e-4, 1e-5, 1e-6]),
            params.Categorical("weight_decay", [1e-3, 1e-4, 1e-5]),
            params.Categorical("dropout", [0.0, 0.1, 0.2, 0.4]),
            # Data Augmentation
            params.Categorical("data_augmentation", data_augmentation_strategies),
        ]

    def _fit(self, X: dict, *args: Any, **kwargs: Any) -> "MetaBlockPlugin":

        X_tab = torch.from_numpy(np.asarray(X[TABULAR_KEY]))
        X_img = X[IMAGE_KEY]

        y = args[0]
        self.n_classes = len(np.unique(y))
        y = torch.from_numpy(np.asarray(y))

        self.model = MetaBlockArchitecture(
            conv_name=self.conv_name,
            n_classes=self.n_classes,
            n_metadata=X_tab.shape[1],
            transform=self.data_augmentation,
            preprocess=self.preprocess,
            n_reducer_layer=self.n_reducer_layer,
            n_reducer_neurons=self.n_reducer_neurons,
            freeze_conv=False,
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
            batch_size=self.batch_size,
            weighted_cross_entropy=self.weighted_cross_entropy,
        )

        self.model.train(X_tab, X_img, y)

        return self

    def _predict_proba(self, X: dict, *args: Any, **kwargs: Any) -> pd.DataFrame:

        self.model.eval()
        self.model.to(DEVICE)

        with torch.no_grad():
            X_img = X[IMAGE_KEY]
            X_tab = torch.from_numpy(np.asarray(X[TABULAR_KEY])).float()
            results = np.empty((0, self.n_classes))
            test_dataset = TestMultimodalDataset(
                X_tab, X_img, weight_transform=self.preprocess
            )
            test_loader = DataLoader(test_dataset, batch_size=100, pin_memory=True)
            for batch_test_ndx, X_test in enumerate(test_loader):
                X_tab, X_img = X_test
                results = np.vstack(
                    (
                        results,
                        nn.Softmax(dim=1)(
                            self.model(X_tab.to(DEVICE), X_img.to(DEVICE))
                        )
                        .detach()
                        .cpu()
                        .numpy(),
                    )
                )
            return pd.DataFrame(results)

    def predict_proba_tensor(self, X: dict):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        # CPU and Eval mode to get access to grad
        self.model.cpu()
        self.model.eval()

        # Store Results
        results = torch.empty((0, self.n_classes))

        # Pass the data point to the model
        X_img = X[IMAGE_KEY]
        X_tab = torch.from_numpy(np.asarray(X[TABULAR_KEY].T)).float()
        test_dataset = TestMultimodalDataset(
            X_tab, X_img, weight_transform=self.preprocess
        )
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        for batch_test_ndx, X_test in enumerate(test_loader):
            X_tab, X_img = X_test
            X_img = X_img.cpu()
            X_tab = X_tab.cpu()
            X_img.requires_grad = True
            results = torch.cat(
                (
                    results,
                    self.model(X_tab, X_img),
                ),
                dim=0,
            )
        return results

    def _predict(self, X: dict, *args: Any, **kwargs: Any) -> pd.DataFrame:

        self.model.eval()
        self.model.to(DEVICE)

        with torch.no_grad():
            X_img = X[IMAGE_KEY]
            X_tab = torch.from_numpy(np.asarray(X[TABULAR_KEY])).float()
            results = np.empty((0, 1))
            test_dataset = TestMultimodalDataset(
                X_tab, X_img, weight_transform=self.preprocess
            )
            test_loader = DataLoader(test_dataset, batch_size=100, pin_memory=False)
            for batch_test_ndx, X_test in enumerate(test_loader):
                X_tab, X_img = X_test
                results = np.vstack(
                    (
                        results,
                        np.expand_dims(
                            self.model.forward(X_tab.to(DEVICE), X_img.to(DEVICE))
                            .argmax(dim=-1)
                            .detach()
                            .cpu()
                            .numpy(),
                            axis=1,
                        ).astype(int),
                    )
                )
            return pd.DataFrame(results)

    def get_image_model(self):
        return self.model.get_image_model()

    def zero_grad(self):
        self.model.zero_grad_model()

    def get_conv_name(self):
        return self.conv_name

    def get_size(self):
        return models.get_weight(WEIGHTS[self.conv_name]).transforms.keywords[
            "crop_size"
        ]

    def save(self) -> bytes:
        return save_model(self.model)

    @classmethod
    def load(cls, buff: bytes) -> "MetaBlockPlugin":
        model = load_model(buff)

        return cls(model=model)


plugin = MetaBlockPlugin
