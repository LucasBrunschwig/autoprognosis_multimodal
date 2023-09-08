# stdlib
import time
from typing import Any, List, Optional

# third party
import numpy as np
import pandas as pd

# autoprognosis absolute
from autoprognosis.explorers.core.selector import predefined_args
import autoprognosis.logger as log
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.prediction.classifiers.base as base
from autoprognosis.utils.default_modalities import IMAGE_KEY, TABULAR_KEY
from autoprognosis.utils.distributions import enable_reproducible_results
from autoprognosis.utils.pip import install
from autoprognosis.utils.serialization import load_model, save_model

for retry in range(2):
    try:
        # third party
        import torch
        from torch import nn
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        from torch.utils.data import DataLoader, Dataset

        break
    except ImportError:
        depends = ["torch"]
        install(depends)
for retry in range(2):
    try:
        # third party
        import torchvision
        from torchvision import transforms

        break
    except ImportError:
        depends = ["torchvision"]
        install(depends)

LR = {
    0: [1e-4, 1e-5],
    1: [1e-4, 1e-4],
    2: [1e-5, 1e-6],
    3: [1e-5, 1e-5],
    4: [1e-6, 1e-6],
    5: [1e-6, 1e-7],
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TrainingImageDataset(Dataset):
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


class TrainingDataset(Dataset):
    def __init__(
        self,
        data_tensor_tab: torch.Tensor,
        data_image: pd.DataFrame,
        target_tensor: torch.Tensor,
        weight_transform,
        transform: torchvision.transforms.Compose = None,
    ):
        """
        CustomDataset constructor.

        Args:
        images (torch.Tensor): List of image tensors, where each tensor rows represent an image.
        labels (torch.Tensor): Tensor containing the labels corresponding to the images.
        transform (callable, optional): Optional transformations to be applied to the images. Default is None.
        """
        self.transform = transform
        self.weight_transform = weight_transform
        self.image = data_image.squeeze()
        self.tab = data_tensor_tab
        self.target = target_tensor

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image, tab, label = self.image.iloc[index], self.tab[index], self.target[index]

        if self.transform:
            image = self.transform(image)

        image = self.weight_transform(image)

        return tab, image, label


class TestTensorDataset(Dataset):
    def __init__(self, data_tensor_tab, data_image, weight_transform):
        """
        CustomDataset constructor.

        Args:
        images (torch.Tensor): List of image tensors, where each tensor rows represent an image.
        labels (torch.Tensor): Tensor containing the labels corresponding to the images.
        transform (callable, optional): Optional transformations to be applied to the images. Default is None.
        """
        self.weight_transform = weight_transform
        self.image = data_image.squeeze(axis=1)
        self.tab = data_tensor_tab

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image, tab = self.image.iloc[index], self.tab[index]

        image = self.weight_transform(image)

        return tab, image


class MultimodalArchitecture(nn.Module):
    def __init__(self, feature_vector_dim, num_classes):
        super(MultimodalArchitecture, self).__init__()

        # Image to Patches and Linear Embedding
        self.conv = nn.Conv2d(
            3, 768, kernel_size=16, stride=16
        )  # Convert image to patches
        self.patch_embedding = nn.Linear(768, 768)
        self.class_token = nn.Parameter(torch.randn(1, 1, 768))  # Class token

        # Image Transformer Encoder
        encoder_layers = TransformerEncoderLayer(d_model=768, nhead=8)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=12)

        # Feature Vector Embeddings
        self.soft_embedding = nn.Softmax(dim=1)
        self.linear_embedding = nn.Linear(feature_vector_dim, 768)

        # Multihead Attention
        self.multihead_attn = nn.MultiheadAttention(embed_dim=768, num_heads=8)

        # Classifier
        self.classifier = nn.Linear(768, num_classes)

        self.optimizer = torch.optim.Adam(self.parameters())

    def forward(self, image, feature_vector):
        # Image to Patches and Linear Embedding
        x_image = (
            self.conv(image).flatten(2).permute(2, 0, 1)
        )  # Convert image to patches and permute
        x_image = self.patch_embedding(x_image)
        x_image = torch.cat(
            [self.class_token.repeat(1, image.size(0), 1), x_image], dim=0
        )  # Add class token

        # Image Transformer Encoder
        x_image = self.transformer_encoder(x_image)

        # Feature Vector Embeddings
        x_feature = self.soft_embedding(feature_vector)
        x_feature = self.linear_embedding(x_feature)

        # Concatenate
        x_concat = torch.cat((x_image[-1], x_feature), dim=1)

        # Multihead Attention
        attn_output, _ = self.multihead_attn(x_concat, x_concat, x_concat)
        x = attn_output.mean(dim=0)

        # Classifier
        x = self.classifier(x)

        return x

    def train(
        self, X_tab: torch.Tensor, X_img: pd.DataFrame, y: torch.Tensor
    ) -> "VitAttPlugin":

        X_tab = self._check_tensor(X_tab).float()
        y = self._check_tensor(y).squeeze().long()

        dataset = TrainingDataset(
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
            # prefetch_factor=3,
            # num_workers=5,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            # prefetch_factor=3,
            # num_workers=5,
        )

        # do training
        val_loss_best = 999999
        patience = 0

        # # TMP LUCAS
        # label_counts = torch.bincount(y)
        # class_weights = 1. / label_counts.float()
        # class_weights = class_weights / class_weights.sum()
        # class_weights = class_weights.to(DEVICE)
        # loss = nn.CrossEntropyLoss(weight=class_weights)
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
                            print(
                                f"Epoch: {i}, loss: {val_loss:.4f}, train_loss: {torch.mean(train_loss):.4f}, "
                                f"elapsed time {(end_ - start_):.2f}"
                            )
                            break

                    if i % self.n_iter_print == 0:
                        log.trace(
                            f"Epoch: {i}, loss: {val_loss:.4f}, train_loss: {torch.mean(train_loss):.4f}, "
                            f"elapsed time {(end_ - start_):.2f}"
                        )

        return self


class VitAttPlugin(base.ClassifierPlugin):
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
        # Data Augmentation
        data_augmentation: bool = "simple_strategy",
        transformation: transforms.Compose = None,
        # Training
        lr: int = 3,
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
        self.lr = LR[lr]
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
        self.n_classes = None

        # Data Augmentation
        self.transformation = transformation
        self.preprocess = None
        self.data_augmentation = data_augmentation
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

        return [
            # CNN Architecture
            # Training
            params.Integer("lr", 0, 5),
            # Data Augmentation
            params.Categorical(
                "data_augmentation",
                [
                    "",
                    # "autoaugment_cifar10",
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
                    transforms.GaussianBlur(3, sigma=(0.1, 0.5)),
                ]

            self.transforms_compose = transforms.Compose(self.transforms)

        else:
            self.transforms_compose = None

    @staticmethod
    def to_tensor(img_: pd.DataFrame) -> torch.Tensor:
        img_ = img_.squeeze(axis=1).apply(lambda d: transforms.ToTensor()(d))
        return torch.stack(img_.tolist())

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "VitAttPlugin":
        if len(*args) == 0:
            raise RuntimeError("Please provide the labels for training")

        X_tab = torch.from_numpy(np.asarray(X[TABULAR_KEY]))
        X_img = X[IMAGE_KEY]

        y = args[0]
        self.n_classes = len(np.unique(y))
        y = torch.from_numpy(np.asarray(y))
        n_classes = np.unique(y).shape[0]
        self.n_classes = n_classes

        self.image_transform()

        self.model = MultimodalArchitecture(n_classes=n_classes)

        self.model.train(X_tab, X_img, y)

        return self

    def _predict(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
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
        with torch.no_grad():
            results = np.empty((0, self.n_classes))
            test_dataset = TestTensorDataset(X, preprocess=self.preprocess)
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                prefetch_factor=2,
                num_workers=5,
            )
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
        return self.size

    def save(self) -> bytes:
        return save_model(self)

    @classmethod
    def load(cls, buff: bytes) -> "VitAttPlugin":
        return load_model(buff)


plugin = VitAttPlugin


# Example usage:
model = MultimodalArchitecture(feature_vector_dim=512, num_classes=10)
image = torch.randn(32, 3, 256, 256)  # Batch of 32 images of size 256x256x3
feature_vector = torch.randn(32, 512)  # Batch of 32 feature vectors of size 512
output = model(image, feature_vector)
print(output.shape)  # torch.Size([32, 10])
