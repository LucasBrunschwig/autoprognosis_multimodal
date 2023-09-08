# stdlib
import time
from typing import Any, List

# third party
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import torchvision.transforms

# autoprognosis absolute
from autoprognosis.explorers.core.defaults import CNN as PREDEFINED_CNN, WEIGHTS
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
        from torch.utils.data import DataLoader, Dataset, Subset

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

        if isinstance(image, pd.Series):
            image = image.squeeze()

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


class MetaBlock(nn.Module):
    """
    Implementing the Metadata Processing Block (MetaBlock)
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
    def __init__(
        self,
        base_model,
        num_class,
        n_metadata,
        conv_name: str,
        transform: transforms.Compose,
        preprocess,
        n_reducer_neurons: int = 256,
        n_reducer_layer: int = 1,
        freeze_conv=False,
        dropout: float = 0.5,
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
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        self.n_reducer_layer = n_reducer_layer
        self.n_reducer_neurons = n_reducer_neurons
        self.n_reducer_layer = n_reducer_layer
        self.batch_norm = batch_norm
        self.preprocess = preprocess
        self.dropout = dropout
        self.transform = transform
        self.base_model = base_model
        self.batch_size = batch_size

        # Training
        self.lr = lr
        self.weight_decay = weight_decay
        self.clipping_value = clipping_value
        self.n_iter = n_iter
        self.n_iter_min = n_iter_min
        self.n_iter_print = n_iter_print
        self.patience = patience
        self.early_stopping = early_stopping

        n_feat_conv = self.feature_maps_size()
        self.n_feat_conv = n_feat_conv

        if conv_name == "alexnet":
            self.comb_feat_maps = 256
            self.feat_map_size = 36
        elif conv_name == "vgg13":
            self.feat_map_size = 512
            self.comb_feat_maps = 49

        self.combination = MetaBlock(self.comb_feat_maps, n_metadata)

        self.features = nn.Sequential(*list(base_model.children())[:-1])

        # freezing the convolution layers
        if freeze_conv:
            for param in self.features.parameters():
                param.requires_grad = False

        # Feature reducer base
        additional_layers = []
        if n_reducer_layer > 0:
            additional_layers = [
                nn.Linear(n_feat_conv, n_reducer_neurons),
                nn.BatchNorm1d(n_reducer_neurons),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            ]
        if n_reducer_layer > 1:
            for i in range(n_reducer_layer - 1):
                additional_layers.append(
                    nn.Linear(n_reducer_neurons, n_reducer_neurons // 2)
                )
                additional_layers.append(nn.BatchNorm1d(n_reducer_neurons // 2))
                additional_layers.append(nn.ReLU())
                additional_layers.append(nn.Dropout(p=dropout))
                n_reducer_neurons = n_reducer_neurons // 2

        if n_reducer_layer > 0:
            self.reducer_block = nn.Sequential(*additional_layers)
        else:
            self.reducer_block = None

        # Here comes the extra information (if applicable)
        if n_reducer_layer > 0:
            self.classifier = nn.Linear(n_reducer_neurons, num_class)
        else:
            self.classifier = nn.Linear(n_feat_conv, num_class)

        self.features.to(DEVICE)
        self.combination.to(DEVICE)
        if self.reducer_block:
            self.reducer_block.to(DEVICE)
        self.classifier.to(DEVICE)

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

    def feature_maps_size(
        self,
    ):
        features = self.base_model.features

        # dummy image input
        dummy_input = torch.randn(1, 3, 224, 224)

        # Pass the dummy input through the feature extractor
        output = features(dummy_input)

        # Calculate the size of the flattened feature maps
        return output.view(output.size(0), -1).size(1)

    def forward(self, meta_data, img):

        x = self.features(img)
        x = x.view(x.size(0), self.comb_feat_maps, self.feat_map_size, -1).squeeze(
            -1
        )  # getting the feature maps
        x = self.combination(x, meta_data.float())  # applying MetaBlock
        x = x.view(x.size(0), -1)  # flatting
        if self.reducer_block:
            x = self.reducer_block(x)  # feat reducer block

        return self.classifier(x)

    def train(
        self, X_tab: torch.Tensor, X_img: pd.DataFrame, y: torch.Tensor
    ) -> "MetaBlockArchitecture":

        X_tab = self._check_tensor(X_tab).float()
        y = self._check_tensor(y).squeeze().long()

        dataset = TrainingDataset(
            X_tab, X_img, y, weight_transform=self.preprocess, transform=self.transform
        )

        # Using StratifiedShuffleSplit to get stratified indices
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_indices, test_indices = next(sss.split(X_tab, y))

        # Using the indices to create stratified train and test datasets
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

    def _check_tensor(self, X: torch.Tensor) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            return X.cpu()
        else:
            return torch.from_numpy(np.asarray(X)).cpu()

    def get_image_model(self):
        return self.base_model

    def zero_grad_model(self):
        self.base_model.zero_grad()

    def eval_(self):
        self.base_model.eval()
        if self.reducer_block:
            self.reducer_block.eval()
        self.combination.eval()
        self.classifier.eval()

    def train_(self):
        self.base_model.train()
        if self.reducer_block:
            self.reducer_block.train()
        self.combination.train()
        self.classifier.train()


class MetaBlockPlugin(base.ClassifierPlugin):
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
        n_reducer_layer: int = 1,
        n_reducer_neurons: int = 1024,
        conv_name: str = "vgg13",
        dropout: float = 0.4,
        # Training
        data_augmentation: str = "simple_strategy",
        # n_unfrozen_layers: int = 3,
        lr: float = 1e-4,
        weight_decay: float = 1e-3,
        clipping_value: int = 1,
        random_state: int = 0,
        n_iter_print: int = 1,
        patience: int = 5,
        n_iter_min: int = 10,
        n_iter: int = 500,
        batch_norm: bool = True,
        early_stopping: bool = True,
        batch_size: int = 64,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        enable_reproducible_results(random_state)

        self.n_reducer_neurons = n_reducer_neurons
        self.n_reducer_layer = n_reducer_layer

        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.clipping_value = clipping_value
        self.conv_name = conv_name.lower()
        self.data_augmentation = data_augmentation
        self.image_transformation = None
        self.batch_size = batch_size

        weights = models.get_weight(WEIGHTS[self.conv_name.lower()])

        self.base_model = models.get_model(
            conv_name.lower(), weights=WEIGHTS[conv_name.lower()]
        )
        self.preprocess = weights.transforms(antialias=True)

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
            params.Categorical(
                "data_augmentation",
                [
                    "",
                    # "autoaugment_cifar10",
                    "autoaugment_imagenet",
                    "rand_augment",
                    "trivial_augment",
                    "simple_strategy",
                    "gaussian_noise",
                    "color_jittering",
                ],
            ),
        ]

    def image_tranform(self):
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
            elif self.data_augmentation == "simple_strategy":
                self.transforms = [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomResizedCrop(
                        224
                    ),  # Assuming input images are larger than 224x224
                    transforms.RandomRotation(10),
                ]
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

    def _fit(self, X: dict, *args: Any, **kwargs: Any) -> "MetaBlockPlugin":

        X_tab = torch.from_numpy(np.asarray(X[TABULAR_KEY]))
        y = args[0]
        self.n_classes = len(np.unique(y))
        y = torch.from_numpy(np.asarray(y))

        self.image_tranform()

        self.model = MetaBlockArchitecture(
            base_model=self.base_model,
            num_class=self.n_classes,
            n_metadata=X_tab.shape[1],
            transform=self.transforms_compose,
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
            conv_name=self.conv_name,
        )

        X_img = X[IMAGE_KEY]

        # Step 2: fit the newly obtained vector with the selected classifier
        self.model.train(X_tab, X_img, y)

        return self

    def _predict_proba(self, X: dict, *args: Any, **kwargs: Any) -> pd.DataFrame:

        self.model.eval_()
        with torch.no_grad():
            X_img = X[IMAGE_KEY]
            X_tab = torch.from_numpy(np.asarray(X[TABULAR_KEY])).float()
            results = np.empty((0, self.n_classes))
            test_dataset = TestTensorDataset(
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
            self.model.train_()
            return pd.DataFrame(results)

    def _predict(self, X: dict, *args: Any, **kwargs: Any) -> pd.DataFrame:

        self.model.eval_()
        with torch.no_grad():
            X_img = X[IMAGE_KEY]
            X_tab = torch.from_numpy(np.asarray(X[TABULAR_KEY])).float()
            results = np.empty((0, 1))
            test_dataset = TestTensorDataset(
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
            self.model.train_()
            return pd.DataFrame(results)

    def get_image_model(self):
        return self.model.get_image_model()

    def zero_grad(self):
        self.model.zero_grad()

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
