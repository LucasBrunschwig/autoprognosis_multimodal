# stdlib
import time
from typing import Any, List

# third party
import numpy as np
import pandas as pd

# autoprognosis absolute
from autoprognosis.explorers.core.defaults import CNN as PREDEFINED_CNN, WEIGHTS
import autoprognosis.logger as log
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.prediction.classifiers.base as base
from autoprognosis.plugins.prediction.classifiers.tabular.plugin_neural_nets import (
    NONLIN,
)
from autoprognosis.plugins.utils.custom_dataset import (
    TestMultimodalDataset,
    TrainingImageDataset,
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


class ConvIntermediateNet(nn.Module):
    """
    Joint intermediate classifier network for medical images and tabular features.

    Parameters
    ----------
    n_classes: int,
        the final number of classes
    n_tab_in: int,
        number of inputs for tabular features
    n_tab_out: int,
        output size for the tabular sub-model
    preprocess:
        preprocessing for images
    transform:
        data augmentation transform for images
    conv_name: str,
        name of the predefined CNN
    n_tab_layer: int,
        number of layer in the tabular sub-model
    n_img_layer: int,
        number of layer in the image sub-model
    n_img_out: int
        output size of the image model before the final classifier
    n_tab_hidden: int,
        number of hidden neurons in layers of the tabular sub-model
    n_cls_layer: int,
        number of layer in the classifier sub-model
    n_cls_hidden: int,
        number of hidden neurons in layers of the classifier sub-model
    replace_classifier: bool,
        replace the classifier with new layers or add them on top of it
    n_unfrozen_layer: int,
        number of unfrozen layer in the image model
    nonlin: str
        non-linear function in hidden layers
    dropout: float,
        dropout rate
    batch_norm: bool,
        add batch normalization layer
    weighted_cross_entropy: bool,
        use weighted cross-entropy as a loss
    lr: float,
        learning rate
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


    """

    def __init__(
        self,
        n_classes: int,
        n_tab_in: int,
        n_tab_out: int,
        conv_name: str,
        preprocess,
        transform: transforms.Compose = None,
        n_img_layer: int = 2,
        n_img_out: int = 300,
        n_tab_layer: int = 2,
        n_tab_hidden: int = 50,
        n_cls_layer: int = 2,
        n_cls_hidden: int = 100,
        replace_classifier: bool = False,
        n_unfrozen_layer: int = 2,
        nonlin: str = "relu",
        dropout: float = 0.4,
        batch_norm: bool = False,
        weighted_cross_entropy: bool = False,
        lr: float = 1e-4,
        weight_decay: float = 1e-3,
        n_iter: int = 1000,
        batch_size: int = 100,
        n_iter_print: int = 10,
        patience: int = 10,
        n_iter_min: int = 100,
        clipping_value: int = 1,
        early_stopping: bool = True,
    ) -> None:
        super(ConvIntermediateNet, self).__init__()

        if nonlin not in list(NONLIN.keys()):
            raise ValueError("Unknown non-linearity")

        NL = NONLIN[nonlin]

        self.NL = NL
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.preprocess = preprocess
        self.n_classes = n_classes
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.n_iter_print = n_iter_print
        self.patience = patience
        self.n_iter_min = n_iter_min
        self.clipping_value = clipping_value
        self.early_stopping = early_stopping
        self.transform = transform
        self.lr = lr
        self.weighted_cross_entropy = weighted_cross_entropy

        params = []

        self.n_tab_in = n_tab_in
        self.n_tab_out = n_tab_out
        self.n_tab_layer = n_tab_layer
        self.n_tab_hidden = n_tab_hidden

        self.tab_model = self.build_tab_sub_model().to(DEVICE)

        for name, param in self.tab_model.named_parameters():
            params.append({"params": param, "lr": lr, "weight_decay": weight_decay})

        self.replace_classifier = replace_classifier
        self.n_unfrozen_layer = n_unfrozen_layer
        self.conv_name = conv_name.lower()
        self.n_img_layer = n_img_layer
        self.n_img_out = n_img_out

        self.image_model, name_match = self.build_image_sub_model()
        self.image_model = self.image_model.to(DEVICE)

        for name, param in self.image_model.named_parameters():
            if name_match in name:
                param.requires_grad = True
                params.append({"params": param, "lr": lr, "weight_decay": weight_decay})
            elif param.requires_grad:
                params.append(
                    {"params": param, "lr": lr / 10, "weight_decay": weight_decay}
                )

        self.n_unit_in = self.n_tab_out + self.n_img_out
        self.n_cls_hidden = n_cls_hidden
        self.n_cls_layer = n_cls_layer

        self.classifier_model = self.build_classifier_sub_model().to(DEVICE)

        for name, param in self.classifier_model.named_parameters():
            params.append({"params": param, "lr": lr, "weight_decay": weight_decay})

        self.optimizer = torch.optim.Adam(params)

    def build_tab_sub_model(self):
        """Build the tabular sub model network"""

        if self.n_tab_layer > 0:
            if self.batch_norm:
                tab_layer = [
                    nn.Linear(self.n_tab_in, self.n_tab_hidden),
                    nn.BatchNorm1d(self.n_tab_hidden),
                ]
            else:
                tab_layer = [nn.Linear(self.n_tab_in, self.n_tab_hidden)]

            for i in range(self.n_tab_layer - 1):
                if self.batch_norm:
                    tab_layer.extend(
                        [
                            nn.Linear(self.n_tab_hidden, self.n_tab_hidden),
                            nn.BatchNorm1d(self.n_tab_hidden),
                            self.NL(),
                            nn.Dropout(self.dropout),
                        ]
                    )
                else:
                    tab_layer.extend(
                        [
                            nn.Linear(self.n_tab_hidden, self.n_tab_hidden),
                            self.NL(),
                            nn.Dropout(self.dropout),
                        ]
                    )
                    # add final layers
            tab_layer.append(nn.Linear(self.n_tab_hidden, self.n_tab_out))
            tab_layer.append(nn.BatchNorm1d(self.n_tab_out))

        else:
            tab_layer = [nn.Identity()]

        return nn.Sequential(*tab_layer)

    def build_image_sub_model(self):
        """Build the image sub model network"""

        self.image_model = models.get_model(
            self.conv_name, weights=WEIGHTS[self.conv_name]
        )

        # First, freeze all parameters
        for param in self.image_model.parameters():
            param.requires_grad = False

        # Unfroze specified layer
        n_unfrozen_layer = self.unfreeze_last_n_layers_classifier(self.n_unfrozen_layer)
        if n_unfrozen_layer > 0:
            self.unfreeze_last_n_layers_convolutional(n_unfrozen_layer)

        n_features_in = self.extract_n_features_in()

        # The first intermediate layer depends on the last output
        n_intermediate = int((n_features_in // 2))

        additional_layers = []
        if self.n_img_layer > 0:
            additional_layers = [
                nn.Linear(n_features_in, n_intermediate),
                self.NL(),
                nn.Dropout(p=self.dropout, inplace=False),
            ]

            for i in range(self.n_img_layer - 1):
                additional_layers.extend(
                    [
                        nn.Linear(n_intermediate, int(n_intermediate / 2)),
                        self.NL(),
                        nn.Dropout(p=self.dropout, inplace=False),
                    ]
                )
                n_intermediate = int(n_intermediate / 2)

            additional_layers.append(nn.Linear(n_intermediate, self.n_img_out))
        else:
            additional_layers.append(nn.Linear(n_features_in, self.n_img_out))

        additional_layers.append(nn.BatchNorm1d(self.n_img_out))

        name_match = None
        if hasattr(self.image_model, "fc"):
            if (
                isinstance(self.image_model.fc, torch.nn.Sequential)
                and not self.replace_classifier
            ):
                self.image_model.fc[-1] = nn.Sequential(*additional_layers)
                name_match = "fc." + str(len(self.image_model.fc) - 1)
            else:
                self.image_model.fc = nn.Sequential(*additional_layers)
                name_match = "fc"

        elif hasattr(self.image_model, "classifier"):
            if (
                isinstance(self.image_model.classifier, torch.nn.Sequential)
                and not self.replace_classifier
            ):
                self.image_model.classifier[-1] = nn.Sequential(*additional_layers)
                name_match = "classifier." + str(len(self.image_model.classifier) - 1)
            else:
                self.image_model.classifier = nn.Sequential(*additional_layers)
                name_match = "classifier"

        if name_match is None:
            raise ValueError("Unsupported Architecture")

        return self.image_model, name_match

    def extract_n_features_in(self):
        """Extract the size of the input features to define the additional layers."""

        def get_in_features_from_classifier(module, replace_classifier_=False):
            """Helper function to find the first or last linear input size of the classifier"""
            n_features_in = None
            if isinstance(module, torch.nn.Sequential):
                for layer in module:
                    if isinstance(layer, torch.nn.modules.linear.Linear):
                        n_features_in = layer.in_features
                        if replace_classifier_:  # return the first linear layer size
                            break
            elif isinstance(module, torch.nn.modules.linear.Linear):
                n_features_in = module.in_features
            else:
                raise ValueError(f"Unknown Classifier Architecture {self.model_name}")

            return n_features_in

        if hasattr(self.image_model, "fc"):
            classifier = self.image_model.fc
        elif hasattr(self.image_model, "classifier"):
            classifier = self.image_model.classifier
        else:
            raise ValueError(f"Unknown Classifier Module Name: {self.model_name}")

        return get_in_features_from_classifier(classifier, self.replace_classifier)

    def extract_n_features_out(self):
        """Extract the size of the input features to define the additional layers."""

        def get_out_features_from_classifier(module):
            """Helper function to find the first or last linear input size of the classifier"""
            n_features_in = None
            if isinstance(module, torch.nn.Sequential):
                for layer in module:
                    if isinstance(layer, torch.nn.modules.linear.Linear):
                        n_features_in = layer.out_features

            elif isinstance(module, torch.nn.modules.linear.Linear):
                n_features_in = module.out_features
            else:
                raise ValueError(f"Unknown Classifier Architecture {self.model_name}")

            return n_features_in

        if hasattr(self.image_model, "fc"):
            classifier = self.image_model.fc
        elif hasattr(self.image_model, "classifier"):
            classifier = self.image_model.classifier
        else:
            raise ValueError(f"Unknown Classifier Module Name: {self.model_name}")

        return get_out_features_from_classifier(classifier)

    def build_classifier_sub_model(self):
        """Build the classifier sub model network"""

        if self.n_cls_hidden > 0:
            if self.batch_norm:
                layers = [
                    nn.Linear(self.n_unit_in, self.n_cls_hidden),
                    nn.BatchNorm1d(self.n_cls_hidden),
                    self.NL(),
                ]
            else:
                layers = [nn.Linear(self.n_unit_in, self.n_cls_hidden), self.NL()]

            # add required number of layers
            for i in range(self.n_cls_layer - 1):
                if self.batch_norm:
                    layers.extend(
                        [
                            nn.Linear(self.n_cls_hidden, self.n_cls_hidden),
                            nn.BatchNorm1d(self.n_cls_hidden),
                            self.NL(),
                            nn.Dropout(self.dropout),
                        ]
                    )
                else:
                    layers.extend(
                        [
                            nn.Linear(self.n_cls_hidden, self.n_cls_hidden),
                            self.NL(),
                            nn.Dropout(self.dropout),
                        ]
                    )

            # add final layers
            layers.append(nn.Linear(self.n_cls_hidden, self.n_classes))
        else:
            layers = [nn.Linear(self.n_unit_in, self.n_classes)]

        return nn.Sequential(*layers)

    def forward(self, X_tab, X_img) -> torch.Tensor:
        x_tab = self.tab_model(X_tab)
        x_img = self.image_model(X_img)
        x_combined = torch.cat((x_tab, x_img), dim=1)
        return self.classifier_model(x_combined)

    def add_classification_layer(self, n_classes: int):
        """Add a classification layer to the image model for pretraining"""

        n_features_out = self.extract_n_features_out()

        self.image_model.classification = nn.Linear(n_features_out, n_classes)
        self.image_model.classification.to(DEVICE)

    def remove_classification_layer(self):
        """Remove the image model classification layer for whole model training"""
        if hasattr(self.image_model, "classification"):
            self.image_model.classification.cpu()
            del self.image_model.classification

    def forward_image(self, X_img: torch.Tensor):
        output = self.image_model(X_img)
        return self.image_model.classification(output)

    def pretrain_image_model(
        self, X_img: pd.DataFrame, y: torch.Tensor, n_classes: int
    ):
        """Fine-Tuning the image model on images only before fitting all sub-models"""

        y = self._check_tensor(y).squeeze().long()

        self.add_classification_layer(n_classes)

        optimizer_image = torch.optim.Adam(
            params=self.image_model.parameters(), lr=self.lr
        )

        dataset = TrainingImageDataset(
            X_img, y, preprocess=self.preprocess, transform=self.transform
        )

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size]
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
        )

        val_loss_best = 999999
        patience = 0

        if self.weighted_cross_entropy:
            label_counts = torch.bincount(y)
            class_weights = 1.0 / label_counts.float()
            class_weights = class_weights / class_weights.sum()
            class_weights = class_weights.to(DEVICE)
            loss = nn.CrossEntropyLoss(weight=class_weights)
        else:
            loss = nn.CrossEntropyLoss()

        self.train()
        n_iter = min(self.n_iter, 50)
        for i in range(n_iter):
            train_loss = []

            start_ = time.time()

            for batch_ndx, sample in enumerate(train_loader):
                optimizer_image.zero_grad()

                X_img_next, y_next = sample
                if torch.cuda.is_available():
                    X_img_next = X_img_next.to(DEVICE)
                    y_next = y_next.to(DEVICE)

                preds = self.forward_image(X_img_next).squeeze()

                batch_loss = loss(preds, y_next)

                batch_loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    list(self.classifier_model.parameters())
                    + list(self.image_model.parameters())
                    + list(self.tab_model.parameters()),
                    self.clipping_value,
                )

                optimizer_image.step()

                train_loss.append(batch_loss.detach())

            train_loss = torch.Tensor(train_loss).to(DEVICE)
            end_ = time.time()

            if self.early_stopping or i % self.n_iter_print == 0:
                with torch.no_grad():
                    val_loss = []
                    for batch_test_ndx, val_sample in enumerate(test_loader):
                        X_img_val, y_val = val_sample

                        if torch.cuda.is_available():
                            X_img_val = X_img_val.to(DEVICE)
                            y_val = y_val.to(DEVICE)

                        preds = self.forward_image(X_img_val).squeeze()
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
                                f"Pretraining - Epoch: {i}, loss: {val_loss:.4f}, train_loss: {torch.mean(train_loss):.4f}, "
                                f"elapsed time {(end_ - start_):.2f}"
                            )
                            break

                    if i % self.n_iter_print == 0:
                        log.trace(
                            f"Pretraining - Epoch: {i}, loss: {val_loss:.4f}, train_loss: {torch.mean(train_loss):.4f}, "
                            f"elapsed time {(end_-start_):.2f}"
                        )

        self.remove_classification_layer()

        return self

    def train_(
        self, X_tab: torch.Tensor, X_img: pd.DataFrame, y: torch.Tensor
    ) -> "ConvIntermediateNet":

        X_tab = self._check_tensor(X_tab).float()
        y = self._check_tensor(y).squeeze().long()

        dataset = TrainingMultimodalDataset(
            X_tab, X_img, y, weight_transform=self.preprocess, transform=self.transform
        )

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size]
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
        )

        val_loss_best = 999999
        patience = 0

        if self.weighted_cross_entropy:
            label_counts = torch.bincount(y)
            class_weights = 1.0 / label_counts.float()
            class_weights = class_weights / class_weights.sum()
            class_weights = class_weights.to(DEVICE)
            loss = nn.CrossEntropyLoss(weight=class_weights)
        else:
            loss = nn.CrossEntropyLoss()

        self.train()
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

                torch.nn.utils.clip_grad_norm_(
                    list(self.classifier_model.parameters())
                    + list(self.image_model.parameters())
                    + list(self.tab_model.parameters()),
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
                                f"Epoch: {i}, loss: {val_loss:.4f}, train_loss: {torch.mean(train_loss):.4f}, "
                                f"elapsed time {(end_ - start_):.2f}"
                            )
                            break

                    if i % self.n_iter_print == 0:
                        log.trace(
                            f"Epoch: {i}, loss: {val_loss:.4f}, train_loss: {torch.mean(train_loss):.4f}, "
                            f"elapsed time {(end_-start_):.2f}"
                        )

        return self

    def _check_tensor(self, X: torch.Tensor) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            return X.cpu()
        else:
            return torch.from_numpy(np.asarray(X)).cpu()

    def unfreeze_last_n_layers_convolutional(self, n):

        # Count the unfrozen layers
        unfrozen_count = 0

        # Define a recursive function for depth-first search
        def dfs(module):
            nonlocal unfrozen_count
            for child in reversed(list(module.children())):
                dfs(child)
                if unfrozen_count >= n:
                    return
                if isinstance(
                    child, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.BatchNorm2d)
                ):
                    for param in child.parameters():
                        param.requires_grad = True
                    unfrozen_count += 1

        # Select the feature extraction part of the model
        if isinstance(self.image_model, torch.nn.Sequential):
            feature_extractor = self.image_model
        elif hasattr(self.image_model, "features"):
            feature_extractor = self.image_model.features
        elif hasattr(self.image_model, "layer4"):  # For ResNet
            feature_extractor = self.image_model.layer4
        else:
            raise ValueError("Unsupported architecture")

        # Perform depth-first search on the feature extractor
        dfs(feature_extractor)

    def unfreeze_last_n_layers_classifier(self, n):

        n = n + 1  # last linear layer is replaced by the new classifier

        # Count the unfrozen layers
        unfrozen_count = 0

        # Define a recursive function for depth-first search
        def dfs(module):
            nonlocal unfrozen_count
            for child in reversed(list(module.children())):
                dfs(child)
                if unfrozen_count >= n:
                    return
                if isinstance(child, (torch.nn.Linear, torch.nn.BatchNorm1d)):
                    for param in child.parameters():
                        param.requires_grad = True
                    unfrozen_count += 1

        # Select the feature extraction part of the model
        if isinstance(self.image_model, torch.nn.Sequential):
            feature_extractor = self.image_model
        elif hasattr(self.image_model, "classifier"):
            feature_extractor = self.image_model.classifier
        elif hasattr(self.image_model, "fc"):  # For ResNet
            feature_extractor = self.image_model.fc
        else:
            raise ValueError("Unsupported architecture")

        # Perform depth-first search on the feature extractor
        dfs(feature_extractor)

        return n - unfrozen_count

    def get_image_model(self):
        return self.image_model

    def get_tabular_model(self):
        return self.tab_model


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
        # Network Architecture
        nonlin="relu",
        n_tab_layer: int = 0,
        tab_reduction_ratio=3.0,
        n_img_layer: int = 2,
        conv_name: str = "alexnet",
        n_tab_hidden: int = 64,
        n_img_out: int = 50,
        n_cls_layers: int = 1,
        n_cls_hidden: int = 100,
        dropout: float = 0.4,
        # Training
        data_augmentation: str = "simple_strategy",
        n_unfrozen_layers: int = 3,
        lr: float = 1e-4,
        weight_decay: float = 1e-3,
        clipping_value: int = 1,
        random_state: int = 0,
        n_iter_print: int = 1,
        patience: int = 5,
        n_iter_min: int = 10,
        n_iter: int = 1000,
        batch_norm: bool = True,
        early_stopping: bool = True,
        pretrain_image_model: bool = False,
        replace_classifier: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        enable_reproducible_results(random_state)

        # Architecture
        self.conv_name = conv_name.lower()
        self.n_tab_layer = n_tab_layer
        self.n_img_layer = n_img_layer
        self.n_img_out = n_img_out
        self.n_tab_hidden = n_tab_hidden
        self.tab_reduction_ratio = tab_reduction_ratio
        self.n_cls_layer = n_cls_layers
        self.n_cls_hidden = n_cls_hidden
        self.non_linear = nonlin

        # Training
        self.lr = lr
        self.n_unfrozen_layers = n_unfrozen_layers
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.clipping_value = clipping_value
        self.pretrain_image_model = pretrain_image_model
        self.replace_classifier = replace_classifier
        self.batch_norm = batch_norm
        self.early_stopping = early_stopping
        self.patience = patience
        self.n_iter = n_iter

        # Preprocessing
        self.preprocess = self.image_preprocessing()
        self.data_augmentation = build_data_augmentation_strategy(data_augmentation)

        # Miscellaneous
        self.n_iter_print = n_iter_print
        self.n_iter_min = n_iter_min

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
        if kwargs.get("predefined_cnn", None) and len(kwargs["predefined_cnn"]) > 0:
            CNN = kwargs["predefined_cnn"]
        else:
            CNN = PREDEFINED_CNN

        return [
            # Network for Tabular and Image network
            params.Categorical("tab_reduction_ratio", [1.0, 2.0, 4.0]),
            params.Integer("n_img_out", 25, 175),
            params.Integer("n_tab_layer", 0, 4),
            params.Integer("n_img_layer", 0, 4),
            params.Categorical("conv_name", CNN),
            # Final Classifiers
            params.Integer("n_layers_hidden", 1, 4),
            params.Integer("n_units_hidden", 50, 100),
            # Training and global parameters
            params.Categorical("lr", [1e-4, 1e-5, 1e-6]),
            params.Categorical("weight_decay", [1e-3, 1e-4, 1e-5]),
            params.Categorical("dropout", [0.0, 0.1, 0.2, 0.4]),
            params.Integer("n_unfrozen_layers", 1, 8),
            params.Categorical("pretrain_image_model", [True, False]),
            params.Categorical("replace_classifier", [True, False]),
            # Data Augmentation
            params.Categorical("data_augmentation", data_augmentation_strategies),
        ]

    def image_preprocessing(self):
        weights = models.get_weight(WEIGHTS[self.conv_name.lower()])
        return weights.transforms(antialias=True)

    def _fit(
        self, X: dict, *args: Any, **kwargs: Any
    ) -> "IntermediateFusionConvNetPlugin":

        X_tab = torch.from_numpy(np.asarray(X[TABULAR_KEY]))
        X_img = X[IMAGE_KEY]
        if isinstance(X_img, np.ndarray):
            X_img = pd.DataFrame(X_img)

        y = args[0]
        self.n_classes = len(np.unique(y))
        y = torch.from_numpy(np.asarray(y))

        n_tab_out = int(self.tab_reduction_ratio * X_tab.shape[1])
        if self.n_tab_layer == 0:
            n_tab_out = X_tab.shape[1]

        # TODO cutoff for the number of X_tab ? -> force n_tab_layer = 0 if
        self.model = ConvIntermediateNet(
            n_classes=self.n_classes,
            n_tab_in=X_tab.shape[1],
            n_tab_out=n_tab_out,
            n_tab_layer=self.n_tab_layer,
            n_tab_hidden=self.n_tab_hidden,
            conv_name=self.conv_name,
            n_img_layer=self.n_img_layer,
            n_img_out=self.n_img_out,
            n_cls_hidden=self.n_cls_hidden,
            n_cls_layer=self.n_cls_layer,
            n_unfrozen_layer=self.n_unfrozen_layers,
            transform=self.data_augmentation,
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
            preprocess=self.preprocess,
            replace_classifier=self.replace_classifier,
        )

        if self.pretrain_image_model:
            self.model.pretrain_image_model(X_img, y, self.n_classes)

        self.model.train_(X_tab, X_img, y)

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
        test_loader = DataLoader(test_dataset, batch_size=100)
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
                            self.model(X_tab.to(DEVICE), X_img.to(DEVICE))
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

    def get_tabular_model(self):
        return self.model.get_tabular_model()

    def zero_grad(self):
        self.model.zero_grad()

    def get_conv_name(self):
        return self.conv_name

    def get_size(self):
        return models.get_weight(WEIGHTS[self.conv_name]).transforms.keywords[
            "crop_size"
        ]

    def save(self) -> bytes:
        return save_model(self.model)

    @classmethod
    def load(cls, buff: bytes) -> "IntermediateFusionConvNetPlugin":
        model = load_model(buff)

        return cls(model=model)


plugin = IntermediateFusionConvNetPlugin
