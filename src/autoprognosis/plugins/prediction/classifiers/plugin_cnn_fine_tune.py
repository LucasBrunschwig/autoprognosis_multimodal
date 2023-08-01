# stdlib
from typing import Any, List, Optional

# third party
import numpy as np
import pandas as pd

# autoprognosis absolute
from autoprognosis.explorers.core.defaults import (
    CNN as PREDEFINED_CNN,
    CNN_MODEL,
    WEIGHTS,
)
from autoprognosis.explorers.core.selector import predefined_args

# import autoprognosis.logger as log
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
        from torchvision import transforms
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


class TrainingTensorDataset(TensorDataset):
    def __init__(self, data_tensor, target_tensor, preprocess, transform=None):
        """
        CustomDataset constructor.

        Args:
        images (torch.Tensor): List of image tensors, where each tensor rows represent an image.
        labels (torch.Tensor): Tensor containing the labels corresponding to the images.
        transform (callable, optional): Optional transformations to be applied to the images. Default is None.
        """
        super(TrainingTensorDataset, self).__init__(data_tensor, target_tensor)
        self.transform = transforms.Compose(transform.transforms)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.tensors[0])

    def __getitem__(self, index):
        image, label = super(TrainingTensorDataset, self).__getitem__(index)

        if self.transform:
            image = self.transform(image)

        image = self.preprocess(image)

        return image, label


class TestTensorDataset(TensorDataset):
    def __init__(self, data_tensor, preprocess):
        """
        CustomDataset constructor.

        Args:
        images (PIL, Images): List of image tensors, where each tensor rows represent an image.
        labels (torch.Tensor): Tensor containing the labels corresponding to the images.
        transform (callable, optional): Optional transformations to be applied to the images. Default is None.
        """
        super(TestTensorDataset, self).__init__(data_tensor)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.tensors[0])

    def __getitem__(self, index):
        image = super(TestTensorDataset, self).__getitem__(index)[0]
        image = self.preprocess(image)

        return image


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
        n_unfrozen_layer: int = 0,
        n_additional_layers: int = 2,
    ):

        super(ConvNetPredefinedFineTune, self).__init__()

        # Training Parameters
        self.transforms = transformation
        self.batch_size = batch_size
        self.lr = lr
        self.n_iter = n_iter
        self.n_iter_min = n_iter_min
        self.early_stopping = early_stopping
        self.n_iter_print = n_iter_print
        self.patience = patience
        self.preprocess = preprocess

        # Model Architectures
        self.model_name = model_name.lower()
        self.n_additional_layers = n_additional_layers
        self.model = CNN_MODEL[self.model_name](weights=WEIGHTS[self.model_name]).to(
            DEVICE
        )

        # Unfroze specified layer
        self.unfreeze_last_n_layers(n_unfrozen_layer)

        if n_classes is None:
            raise RuntimeError(
                "To build the architecture, the CNN requires to know the number of classes"
            )

        # Replace the output layer by the given number of classes
        if hasattr(self.model, "fc"):
            n_features_in = self.model.fc.in_features

        elif hasattr(self.model, "classifier"):
            if isinstance(self.model.classifier, torch.nn.Sequential):
                n_features_in = self.model.classifier[-1].in_features
            else:
                n_features_in = self.model.classifier.in_features
        else:
            raise ValueError(f"Model not implemented: {self.model_name}")

        # The first intermediate layer depends on the last output
        n_intermediate = int((n_features_in // 2))
        NL = NONLIN[non_linear]

        additional_layers = [
            nn.Linear(n_features_in, n_intermediate),
            NL(inplace=True),
        ]

        for i in range(n_additional_layers - 1):
            additional_layers.extend(
                [
                    nn.Linear(n_intermediate, int(n_intermediate / 2)),
                    NL(inplace=True),
                ]
            )
            n_intermediate = int(n_intermediate / 2)

        additional_layers.append(nn.Linear(n_intermediate, n_classes))

        params_ = []
        if hasattr(self.model, "fc"):
            self.model.fc = nn.Sequential(*additional_layers)
            self.model.to(DEVICE)
            for name, param in self.model.named_parameters():
                if "fc" in name:
                    params_.append({"params": param, "lr": lr})
                elif param.requires_grad:
                    params_.append({"params": param, "lr": 1e-5})
        elif hasattr(self.model, "classifier"):
            if isinstance(self.model.classifier, torch.nn.modules.Sequential):
                self.model.classifier[-1] = nn.Sequential(*additional_layers)
            else:
                self.model.classifier = nn.Sequential(*additional_layers)
            name_match = "classifier"
            self.model.to(DEVICE)
            for name, param in self.model.named_parameters():
                if name_match in name:
                    param.requires_grad = True
                    params_.append(
                        {"params": param, "lr": lr, "weight_decay": weight_decay}
                    )
                elif param.requires_grad:
                    params_.append(
                        {"params": param, "lr": 1e-5, "weight_decay": weight_decay}
                    )

        self.optimizer = torch.optim.Adam(params_)

    def unfreeze_last_n_layers(self, n):
        # First, freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

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
        if isinstance(self.model, torch.nn.Sequential):
            feature_extractor = self.model
        elif hasattr(self.model, "features"):
            feature_extractor = self.model.features
        elif hasattr(self.model, "layer4"):  # For ResNet
            feature_extractor = self.model.layer4
        elif hasattr(self.model, "classifier") and hasattr(
            self.model, "features"
        ):  # For MobileNet
            feature_extractor = self.model.features
        elif hasattr(self.model, "features") and hasattr(
            self.model, "classifier"
        ):  # For DenseNet
            feature_extractor = self.model.features
        else:
            raise ValueError("Unsupported architecture")

        # Perform depth-first search on the feature extractor
        dfs(feature_extractor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_model(self):
        return self.model

    def set_zero_grad(self):
        self.model.zero_grad()

    def train(self, X: torch.Tensor, y: torch.Tensor) -> "ConvNetPredefinedFineTune":
        X = self._check_tensor(X).float()
        y = self._check_tensor(y).squeeze().long()

        dataset = TrainingTensorDataset(
            X, y, preprocess=self.preprocess, transform=self.transforms
        )

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        test_size = min(test_size, 300)
        train_size = len(dataset) - test_size

        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size]
        )

        loader = DataLoader(
            train_dataset, batch_size=self.batch_size, prefetch_factor=3, num_workers=10
        )
        val_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
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
                        print(
                            f"Epoch: {i}, loss: {val_loss}, train_loss: {torch.mean(train_loss)}"
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
                self.model.classifier[-1][-1] = nn.Identity()
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


class CNNFineTunePlugin(base.ClassifierPlugin):
    """Classification plugin using predefined Convolutional Neural Networks

    Parameters
    ----------
    conv_net: str,
        the predefined architecture
    n_unfrozen_layers:
        the number of layer to unfreeze
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
        # Architecture
        conv_net: str = "alexnet",
        n_unfrozen_layer: int = 1,
        n_additional_layers: int = 2,
        non_linear: str = "relu",
        # Data Augmentation
        data_augmentation: bool = True,
        color_jittering: bool = True,
        gaussian_noise: bool = True,
        transformation: transforms.Compose = None,
        # Training
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        n_iter: int = 1000,
        batch_size: int = 100,
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

        # Training Parameters
        self.lr = lr
        self.non_linear = non_linear
        self.weight_decay = weight_decay
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.n_iter_print = n_iter_print
        self.patience = patience
        self.n_iter_min = n_iter_min
        self.early_stopping = early_stopping

        # CNN Architecture
        self.conv_net = conv_net
        self.n_unfrozen_layer = n_unfrozen_layer
        self.n_classes = None
        self.n_additional_layers = n_additional_layers

        # Data Augmentation
        self.transformation = transformation
        self.preprocess = None
        self.data_augmentation = data_augmentation
        self.color_jittering = color_jittering
        self.gaussian_noise = gaussian_noise
        # Create the Data Transformation
        self.image_transform()

        if (
            predefined_args.get("predefined_cnn", None)
            and len(predefined_args["predefined_cnn"]) > 0
        ):
            self.conv_net = predefined_args["predefined_cnn"][0]

    @staticmethod
    def name() -> str:
        return "cnn_fine_tune"

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
            params.Categorical("conv_net", CNN),
            params.Integer("n_additional_layers", 1, 3),
            # Training
            params.Categorical("lr", [1e-3, 1e-4, 1e-5]),
            params.Integer("n_unfrozen_layer", 0, 3),
            # Data Augmentation
            params.Categorical("data_augmentation", [True, False]),
            params.Categorical("color_jittering", [True, False]),
            params.Categorical("gaussian_noise", [True, False]),
        ]

    def image_transform(self):
        if self.data_augmentation:
            if self.transformation is None:
                self.transforms = [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(10),
                ]
                if self.color_jittering:
                    self.transforms.append(
                        transforms.ColorJitter(
                            brightness=0.1,
                            contrast=0.1,
                            saturation=0.1,
                            hue=0.05,
                        )
                    )
                if self.gaussian_noise:
                    self.transforms.append(
                        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))
                    )
            else:
                self.transforms = self.transformation
            self.transforms_compose = transforms.Compose(self.transforms)

        else:
            self.transforms_compose = None

        weights = models.get_weight(WEIGHTS[self.conv_net.lower()])
        self.preprocess = weights.transforms(antialias=True)

    @staticmethod
    def to_tensor(img_: pd.DataFrame) -> torch.Tensor:
        img_ = img_.squeeze(axis=1).apply(lambda d: transforms.ToTensor()(d))
        return torch.stack(img_.tolist())

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "CNNFineTunePlugin":
        if len(*args) == 0:
            raise RuntimeError("Please provide the labels for training")

        y = args[0]

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        # Preprocess Data
        n_classes = np.unique(y).shape[0]
        self.n_classes = n_classes
        y = torch.from_numpy(np.asarray(y))

        self.model = ConvNetPredefinedFineTune(
            model_name=self.conv_net,
            n_classes=n_classes,
            n_additional_layers=self.n_additional_layers,
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
            transformation=self.transforms_compose,
            preprocess=self.preprocess,
        )

        X = self.to_tensor(X)

        self.model.train(X, y)

        return self

    def _predict(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        X = self.to_tensor(X)
        self.model.to(DEVICE)
        with torch.no_grad():
            results = np.empty((0, 1))
            test_loader = DataLoader(
                TestTensorDataset(X, preprocess=self.preprocess),
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
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        X = self.to_tensor(X)
        self.model.cpu()
        results = torch.empty((0, self.n_classes))
        test_dataset = TestTensorDataset(X, preprocess=self.preprocess)
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
        X = self.to_tensor(X)
        with torch.no_grad():
            results = np.empty((0, self.n_classes))
            test_dataset = TestTensorDataset(X, preprocess=self.preprocess)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
            for batch_test_ndx, X_test in enumerate(test_loader):
                results = np.vstack(
                    (
                        results,
                        self.model(X_test.to(DEVICE)).detach().cpu().numpy(),
                    )
                )

            return pd.DataFrame(results)

    def set_zero_grad(self):
        self.model.zero_grad()

    def get_image_model(self):
        return self.model.get_model()

    def get_size(self):
        return models.get_weight(WEIGHTS[self.conv_net]).transforms.keywords[
            "crop_size"
        ]

    def save(self) -> bytes:
        return save_model(self)

    @classmethod
    def load(cls, buff: bytes) -> "CNNFineTunePlugin":
        return load_model(buff)


plugin = CNNFineTunePlugin
