# stdlib
from typing import Any, List, Optional

# third party
import numpy as np
import pandas as pd

# autoprognosis absolute
from autoprognosis.explorers.core.defaults import CNN as PREDEFINED_CNN, CNN_MODEL
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
        from torch.utils.data import DataLoader, Dataset

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


class CustomDataset(Dataset):
    def __init__(self, data_list, transform=None):
        if isinstance(data_list, list) and isinstance(data_list[0], list):
            self.images = data_list[0]
            self.label = data_list[1]
        else:
            self.images = data_list
            self.label = None
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)

        if self.label is None:
            return image

        label = self.label[idx]
        return image, label


def initialize_weights(module):
    if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)


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
    - allow user to use predefined weight for a given model.

    """

    def __init__(
        self,
        model_name: str,
        n_classes: Optional[int],
        batch_size: int,
        lr: float,
        n_iter: int,
        weight_decay: float,
        early_stopping: bool,
        n_iter_print: int,
        n_iter_min: int,
        patience: int,
        transformation: transforms.Compose = None,
    ):

        super(ConvNetPredefined, self).__init__()

        self.model_name = model_name.lower()

        self.batch_size = batch_size
        self.lr = lr
        self.n_iter = n_iter
        self.n_iter_min = n_iter_min
        self.early_stopping = early_stopping
        self.n_iter_print = n_iter_print
        self.patience = patience
        self.transforms = transformation

        # Load predefined CNN without weights
        self.model = CNN_MODEL[self.model_name](weights=None)

        # Initialize weights
        self.model.apply(initialize_weights)

        if n_classes is None:
            raise RuntimeError(
                "To build the architecture, the CNN requires to know the number of classes"
            )

        # Replace the output layer by the given number of classes
        if hasattr(self.model, "fc"):
            n_features_in = self.model.fc.in_features
            classification_layer = nn.Linear(n_features_in, n_classes)
            self.model.fc = classification_layer

        elif hasattr(self.model, "classifier"):
            if isinstance(self.model.classifier, torch.nn.Sequential):
                n_features_in = self.model.classifier[-1].in_features
                classification_layer = nn.Linear(n_features_in, n_classes)
                self.model.classifier[-1] = classification_layer

            else:
                n_features_in = self.model.classifier.in_features
                classification_layer = nn.Linear(n_features_in, n_classes)
                self.model.classifier = classification_layer

        self.model.to(DEVICE)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

    def preprocess_images(self, img_: pd.DataFrame) -> torch.Tensor:
        return img_.tolist()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def train(self, X: torch.Tensor, y: torch.Tensor) -> "ConvNetPredefined":
        # X = self._check_tensor(X).float()
        y = self._check_tensor(y).squeeze().long()

        dataset = CustomDataset([X, y], transform=self.transforms)

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        test_size = min(test_size, 300)
        train_size = len(dataset) - test_size

        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size]
        )

        loader = DataLoader(train_dataset, batch_size=self.batch_size, pin_memory=False)
        val_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, pin_memory=False
        )

        # do training
        val_loss_best = 999999
        patience = 0

        loss = nn.CrossEntropyLoss()

        for i in range(self.n_iter):
            train_loss = []

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

    def remove_classification_layer(self):
        if hasattr(self.model, "fc"):
            if isinstance(self.model.fc, torch.nn.Sequential):
                self.model.fc[-1] = nn.Identity()
            else:
                self.model.fc = nn.Identity()
        elif hasattr(self.model, "classifier"):
            if isinstance(self.model.classifier, torch.nn.Sequential):
                self.model.classifier[-1] = nn.Identity()
            elif isinstance(self.model.classifier, torch.nn.Linear):
                self.model.classifier = nn.Identity()
            else:
                raise ValueError(
                    f"Unknown classification layer type - {self.model_name}"
                )


class CNNDiffSizePlugin(base.ClassifierPlugin):
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
        conv_net: str = "alexnet",
        n_classes: Optional[int] = None,
        transformation: transforms.Compose = None,
        normalisation: bool = "channel-wise",
        size: int = 256,
        lr: float = 1e-5,
        weight_decay: float = 1e-4,
        n_iter: int = 1000,
        batch_size: int = 100,
        n_iter_print: int = 1,
        data_augmentation: bool = True,
        patience: int = 10,
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
        self.weight_decay = weight_decay
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.size = size
        self.n_iter_print = n_iter_print
        self.patience = patience
        self.n_iter_min = n_iter_min
        self.early_stopping = early_stopping
        self.normalisation = normalisation
        self.data_augmentation = data_augmentation
        self.mean = None
        self.std = None
        self.transforms = []
        self.transform_predict = []
        self.transformation = transformation

        # Specific to CNN model
        self.conv_net = conv_net
        self.n_classes = n_classes

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
            params.Categorical("conv_net", CNN),
            params.Categorical("lr", [1e-4, 1e-5, 1e-6]),
            params.Categorical("data_augmentation", [True, False]),
            params.Categorical("normalisation", ["channel-wise", "pixel-wise"]),
            params.Categorical("size", [128, 256, 512]),
        ]

    def normalisation_values(self, X: pd.DataFrame):

        if self.normalisation == "channel-wise":
            channels = [0, 0, 0]
            n_pixels = 0
            for img in X.squeeze():
                tmp = transforms.ToTensor()(img)
                n_pixels += tmp.size(0) * tmp.size(1) * tmp.size(2)
                channels_sum = tmp.sum((1, 2))
                for i in range(3):
                    channels[i] += float(channels_sum[i])

            # Compute means along each channel
            self.mean = [
                channels[0] / n_pixels,
                channels[1] / n_pixels,
                channels[2] / n_pixels,
            ]

            std = [0, 0, 0]
            for img in X.squeeze():
                tmp = transforms.ToTensor()(img)
                channels_stds = (tmp - torch.Tensor(self.mean).view(3, 1, 1) ** 2).sum(
                    (1, 2)
                )
                for i in range(3):
                    std[i] += float(channels_stds[i])

            # Compute stds along each channel
            self.std = [
                np.sqrt(std[0] / n_pixels),
                np.sqrt(std[1] / n_pixels),
                np.sqrt(std[2] / n_pixels),
            ]

        elif self.normalisation == "pixel-wise":
            pixels = 0
            for img in X.squeeze():
                tmp = transforms.ToTensor()(img)
                for i in range(3):
                    for value in tmp[i, :, :].flatten().tolist():
                        pixels += value

            self.mean = np.mean(pixels)
            self.std = np.std(pixels)

    def image_transform(self):
        if self.data_augmentation:
            if self.transformation is None:
                self.transforms = [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(10),
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                    ),
                    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
                ]
            else:
                self.transforms = self.transformation
        self.transforms.extend(
            [
                transforms.Resize((self.size, self.size), antialias=True),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        self.transforms_compose = transforms.Compose(self.transforms)

        self.transform_predict = transforms.Compose(
            [
                transforms.Resize((self.size, self.size)),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "CNNDiffSizePlugin":
        if len(*args) == 0:
            raise RuntimeError("Please provide the labels for training")

        y = args[0]

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        self.normalisation_values(X)
        self.image_transform()

        # Preprocess Data
        self.n_classes = np.unique(y).shape[0]
        y = torch.from_numpy(np.asarray(y))
        X = X.squeeze().apply(lambda d: transforms.ToTensor()(d))

        self.model = ConvNetPredefined(
            model_name=self.conv_net,
            n_classes=self.n_classes,
            lr=self.lr,
            n_iter=self.n_iter,
            n_iter_min=self.n_iter_min,
            n_iter_print=self.n_iter_print,
            early_stopping=self.early_stopping,
            patience=self.patience,
            batch_size=self.batch_size,
            weight_decay=self.weight_decay,
            transformation=self.transforms_compose,
        )

        X = self.model.preprocess_images(X)

        self.model.train(X, y)

        return self

    def _predict(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        X = self.model.preprocess_images(X.squeeze(axis=1))

        with torch.no_grad():
            results = np.empty((0, self.n_classes))
            test_loader = DataLoader(
                CustomDataset(X, transform=self.transform_predict),
                batch_size=self.batch_size,
                pin_memory=False,
            )
            for batch_test_ndx, X_test in enumerate(test_loader):
                results = np.vstack(
                    (
                        results,
                        self.model(X_test.to(DEVICE))
                        .argmax(dim=-1)
                        .detach()
                        .cpu()
                        .numpy(),
                    )
                )

            return pd.DataFrame(results)

    def _predict_proba(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> pd.DataFrame:
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        X = self.model.preprocess_images(X.squeeze(axis=1))
        with torch.no_grad():
            results = np.empty((0, self.n_classes))
            test_dataset = CustomDataset(X, transform=self.transform_predict)
            test_loader = DataLoader(
                test_dataset, batch_size=self.batch_size, pin_memory=False
            )
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
    def load(cls, buff: bytes) -> "CNNDiffSizePlugin":
        return load_model(buff)


plugin = CNNDiffSizePlugin
