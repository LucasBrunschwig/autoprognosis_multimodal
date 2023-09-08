# stdlib
import time
from typing import Any, List, Optional

# third party
import numpy as np
import pandas as pd
import timm
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torchvision.transforms

# autoprognosis absolute
import autoprognosis.logger as log

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
        from torch import nn, optim
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


class TrainingDataset(Dataset):
    def __init__(self, data, target, preprocess, transform=None):
        """
        CustomDataset constructor.

        Args:
        images (torch.Tensor): List of image tensors, where each tensor rows represent an image.
        labels (torch.Tensor): Tensor containing the labels corresponding to the images.
        transform (callable, optional): Optional transformations to be applied to the images. Default is None.
        """
        self.image = data.squeeze()
        self.target = target
        self.transform = transform
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image, label = self.image.iloc[index], self.target[index]

        if self.transform:
            image = self.transform(image)

        image = self.preprocess(image)

        return image, label


class TestDataset(Dataset):
    def __init__(self, data, preprocess):
        """
        CustomDataset constructor.

        Args:
        images (PIL, Images): List of image tensors, where each tensor rows represent an image.
        labels (torch.Tensor): Tensor containing the labels corresponding to the images.
        transform (callable, optional): Optional transformations to be applied to the images. Default is None.
        """
        self.image = pd.DataFrame(data).squeeze(axis=1)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image = self.image.iloc[index]
        image = self.preprocess(image)

        return image


def get_vision_transformer(
    model_name: str = "vit_tiny_patch16_224",
    pretrained: bool = True,
    num_classes: int = 1000,
    in_chans: int = 3,
):
    """
    Get a Vision Transformer model from the timm library.

    Args:
    - model_name (str): Name of the Vision Transformer variant.
    - pretrained (bool): Whether to load pretrained weights.
    - num_classes (int): Number of output classes.
    - in_chans (int): Number of input channels.

    Returns:
    - model (nn.Module): Vision Transformer model.
    """
    model = timm.create_model(
        model_name, pretrained=pretrained, num_classes=num_classes, in_chans=in_chans
    )
    return model


class VisionTransformerClassifier:
    def __init__(
        self,
        transforms_: transforms.Compose,
        preprocess_: transforms.Compose,
        model_name="vit_base_patch16_224",
        lr: float = 1e-4,
        n_classes=1000,
        pretrained=True,
        in_chans=3,
        n_unfrozen_layer: int = 2,
        n_iter: int = 1000,
        n_iter_min: int = 10,
        n_iter_print: int = 10,
        early_stopping: bool = True,
        patience: int = 10,
        batch_size: int = 128,
        weight_decay=1e-3,
        clipping_value: int = 1,
        weighted_cross_entropy: bool = False,
    ):
        # Model Architecture
        self.model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=n_classes, in_chans=in_chans
        )
        self.model.to(DEVICE)

        # TODO: possible alteration to MLP classifier

        # Preprocessing
        self.transform = transforms_
        self.preprocess = preprocess_

        # Training Parameters
        self.n_iter = n_iter
        self.n_iter_min = n_iter_min
        self.n_iter_print = n_iter_print
        self.early_stopping = early_stopping
        self.patience = patience
        self.batch_size = batch_size
        self.clipping_value = clipping_value
        self.weighted_cross_entropy = weighted_cross_entropy

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

    def forward(self, x):
        return self.model(x)

    def train(self, X: pd.DataFrame, y: torch.Tensor) -> "VisionTransformerClassifier":

        dataset = TrainingDataset(
            X, y, preprocess=self.preprocess, transform=self.transform
        )

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        test_size = min(test_size, 300)
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
                            print(
                                f"Epoch: {i}, loss: {val_loss:.4f}, train_loss: {torch.mean(train_loss):.4f}, elapsed time: {(end_ - start_):.2f}"
                            )
                            break

                    if i % self.n_iter_print == 0:
                        log.trace(
                            f"Epoch: {i}, loss: {val_loss:.4f}, train_loss: {torch.mean(train_loss):.4f}, elapsed time: {(end_ - start_):.2f}"
                        )

        return self


class VisionTransformerPlugin(base.ClassifierPlugin):
    """Classification plugin using predefined Vision Transformer weights and architecture

    Parameters
    ----------
    model_name: str,
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
        model_name: str = "vit_tiny_patch16_224",
        n_unfrozen_layer: int = 2,
        # Data Augmentation
        data_augmentation: bool = "simple_strategy",
        transformation: transforms.Compose = None,
        # Training
        lr: float = 1e-4,
        weighted_cross_entropy: bool = True,
        weight_decay: float = 1e-4,
        n_iter: int = 1000,
        batch_size: int = 100,
        n_iter_print: int = 10,
        patience: int = 5,
        n_iter_min: int = 10,
        early_stopping: bool = True,
        clipping_value: int = 1,
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
        self.weight_decay = weight_decay
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.n_iter_print = n_iter_print
        self.patience = patience
        self.n_iter_min = n_iter_min
        self.early_stopping = early_stopping
        self.clipping_value = clipping_value
        self.weighted_cross_entropy = weighted_cross_entropy

        # CNN Architecture
        self.model_name = model_name
        self.n_unfrozen_layer = n_unfrozen_layer
        self.n_classes = None

        # Data Augmentation
        self.transformation = transformation
        self.preprocess = None
        self.data_augmentation = data_augmentation

        # Create the Data Transformation
        self.image_transform()

    @staticmethod
    def name() -> str:
        return "vision_transformer"

    @staticmethod
    def modality_type() -> str:
        return IMAGE_KEY

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [
            # CNN Architecture
            params.Categorical(
                "model_name",
                [
                    "vit_tiny_patch16_224",
                    "vit_small_patch16_224",
                    "vit_base_patch16_224",
                ],
            ),
            # vit_base_patch32_224 (diff batch), vit_base_patch16_384 (diff im size),
            # vit_base_patch16_224_in21k (imagenet)
            # vit_resnet50_224_in21k (starts with convolutional layer before the transformer architecture)
            # Training
            params.Categorical("lr", [1e-3, 1e-4, 1e-5]),
            params.Integer("n_unfrozen_layer", 1, 8),
            params.Categorical("weighted_cross_entropy", [True, False]),
            # Data Augmentation
            params.Categorical(
                "data_augmentation",
                [
                    "",
                    "autoaugment_cifar10",
                    "autoaugment_imagenet",
                    "rand_augment",
                    "trivial_augment",
                    "simple_strategy",
                    "color_jittering",
                    "gaussian_noise",
                ],
            ),
            params.Categorical("clipping_value", [0, 1]),
        ]

    def image_transform(self):
        if self.data_augmentation:
            if self.data_augmentation == "autoaugment_imagenet":
                self.transforms = [
                    transforms.AutoAugment(
                        policy=transforms.AutoAugmentPolicy.IMAGENET
                    ),
                ]
            elif self.data_augmentation == "autoaugment_cifar10":
                self.transforms = [
                    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
                ]
            elif self.data_augmentation == "rand_augment":
                self.transforms = [
                    transforms.RandAugment(),
                ]
            elif self.data_augmentation == "trivial_augment":
                self.transforms = [transforms.TrivialAugmentWide()]
            elif self.data_augmentation == "gaussian_noise":
                self.transforms = [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomResizedCrop(
                        224
                    ),  # Assuming input images are larger than 224x224
                    transforms.RandomRotation(10),
                    transforms.GaussianBlur(3, sigma=(0.1, 0.5)),
                ]
            elif self.data_augmentation == "simple_strategy":
                self.transforms = [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomResizedCrop(
                        224
                    ),  # Assuming input images are larger than 224x224
                    transforms.RandomRotation(10),
                ]
            elif self.data_augmentation == "color_jittering":
                self.transforms = [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomResizedCrop(
                        224
                    ),  # Assuming input images are larger than 224x224
                    transforms.RandomRotation(
                        10
                    ),  # Random rotation between -10 and 10 degrees
                    transforms.ColorJitter(
                        brightness=0.05, contrast=0.05, saturation=0.05
                    ),
                ]

            self.transforms_compose = transforms.Compose(self.transforms)

        else:
            self.transforms_compose = None

        self.preprocess = transforms.Compose(
            [
                torchvision.transforms.Normalize(
                    mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
                ),
                torchvision.transforms.Resize((224, 224)),
            ]
        )

    @staticmethod
    def to_tensor(img_: pd.DataFrame) -> torch.Tensor:
        img_ = img_.squeeze(axis=1).apply(lambda d: transforms.ToTensor()(d))
        return torch.stack(img_.tolist())

    def _fit(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> "VisionTransformerPlugin":
        if len(*args) == 0:
            raise RuntimeError("Please provide the labels for training")

        y = args[0]

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        # Preprocess Data
        n_classes = np.unique(y).shape[0]
        self.n_classes = n_classes
        y = torch.from_numpy(np.asarray(y))

        self.model = VisionTransformerClassifier(
            model_name=self.model_name,
            n_classes=n_classes,
            lr=self.lr,
            n_unfrozen_layer=self.n_unfrozen_layer,
            n_iter=self.n_iter,
            n_iter_min=self.n_iter_min,
            n_iter_print=self.n_iter_print,
            early_stopping=self.early_stopping,
            patience=self.patience,
            batch_size=self.batch_size,
            weight_decay=self.weight_decay,
            transforms_=self.transforms_compose,
            preprocess_=self.preprocess,
            clipping_value=self.clipping_value,
            weighted_cross_entropy=self.weighted_cross_entropy,
        )

        self.model.train(X, y)

        return self

    def _predict(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        self.model.to(DEVICE)
        with torch.no_grad():
            results = np.empty((0, 1))
            test_loader = DataLoader(
                TestDataset(X, preprocess=self.preprocess),
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
        self.model.cpu()
        self.model.model.eval()
        results = torch.empty((0, self.n_classes))
        test_dataset = TestDataset(X, preprocess=self.preprocess)
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
        self.model.model.train()
        return results

    def _predict_proba(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> pd.DataFrame:
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        self.model.to(DEVICE)
        self.model.model.eval()
        with torch.no_grad():
            results = np.empty((0, self.n_classes))
            test_dataset = TestDataset(X, preprocess=self.preprocess)
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
            self.model.model.train()
            return pd.DataFrame(results)

    def set_zero_grad(self):
        self.model.zero_grad()

    def get_image_model(self):
        return self.model.get_model()

    def get_size(self):
        return 224

    def save(self) -> bytes:
        return save_model(self)

    @classmethod
    def load(cls, buff: bytes) -> "VisionTransformerPlugin":
        return load_model(buff)


plugin = VisionTransformerPlugin
