# stdlib
from typing import Any, List

# third party
import numpy as np
import pandas as pd

# autoprognosis absolute
from autoprognosis.explorers.core.defaults import CNN, CNN_MODEL
from autoprognosis.explorers.core.selector import predefined_args
import autoprognosis.logger as log
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.preprocessors.base as base
from autoprognosis.utils.default_modalities import IMAGE_KEY
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

        break
    except ImportError:
        depends = ["torchvision"]
        install(depends)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPS = 1e-8


# Define the transformation for data augmentation
transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize(mean=0, std=1),
    ]
)


# Custom dataset class to return both original and modified images
class SiameseDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __getitem__(self, index):
        image = self.base_dataset[index]
        original_image = image
        modified_image = self.transform(image)
        return original_image, modified_image

    def __len__(self):
        return len(self.base_dataset)


class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """

    def __init__(
        self,
        base_encoder,
        lr: float,
        batch_size: int,
        n_iter: int,
        weight_decay: float,
        early_stopping: bool,
        n_additional_layers: int,
        n_iter_print: int,
        n_iter_min: int,
        patience: int,
        n_classes: int,
        output_size: int = 50,
        pred_dim: int = 512,
    ):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()

        self.batch_size = batch_size
        self.lr = lr
        self.n_iter = n_iter
        self.n_iter_min = n_iter_min
        self.early_stopping = early_stopping
        self.n_iter_print = n_iter_print
        self.patience = patience
        self.weight_decay = weight_decay

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder(num_classes=output_size, zero_init_residual=True)

        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]

        additional_layers = []
        for i in range(n_additional_layers):

            additional_layers.extend(
                [
                    nn.Linear(prev_dim, prev_dim // 2, bias=False),
                    nn.BatchNorm1d(prev_dim),
                    nn.ReLU(inplace=True),
                ]
            )

            prev_dim = prev_dim // 2

        additional_layers.append(self.encoder.fc)
        self.encoder.fc = nn.Sequential(
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True),  # first layer
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True),  # second layer
            self.encoder.fc,
            nn.BatchNorm1d(output_size, affine=False),
        )  # output layer
        self.encoder.fc[
            6
        ].bias.requires_grad = False  # hack: not use bias as it is followed by BN
        self.encoder.to(DEVICE)
        # build a 2-layer predictor
        self.predictor = nn.Sequential(
            nn.Linear(output_size, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),  # hidden layer
            nn.Linear(pred_dim, n_classes),
        )  # output layer
        self.predictor.to(DEVICE)
        self.optimizer = torch.optim.Adam(
            params=list(self.encoder.parameters()) + list(self.predictor.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        z1 = self.encoder(x1)  # NxC
        z2 = self.encoder(x2)  # NxC

        p1 = self.predictor(z1)  # NxC
        p2 = self.predictor(z2)  # NxC

        return p1, p2, z1.detach(), z2.detach()

    def predict(self, X: torch.Tensor):
        z = self.encoder(X)
        p = self.predictor(z)
        return p

    def train(self, X: torch.Tensor):

        dataset = SiameseDataset(X, transform)

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        test_size = min(test_size, 300)
        train_size = len(dataset) - test_size

        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size]
        )

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, pin_memory=False
        )
        val_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, pin_memory=False
        )

        criterion = nn.CosineSimilarity(dim=1).to(DEVICE)

        # do training
        val_loss_best = 999999
        patience = 0

        for i in range(self.n_iter):
            train_loss = []
            for bach_ndx, images in enumerate(train_loader):

                img1, img2 = images

                if torch.cuda.is_available():
                    img1 = img1.to(DEVICE)
                    img2 = img2.to(DEVICE)

                # compute output and loss
                p1, p2, z1, z2 = self.forward(x1=img1, x2=img2)
                batch_loss = (
                    -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
                )

                # compute gradient and do SGD step
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

                train_loss.append(batch_loss.detach())

            train_loss = torch.Tensor(train_loss).to(DEVICE)

            if self.early_stopping or i % self.n_iter_print == 0:
                with torch.no_grad():

                    val_loss = []

                    for batch_val_ndx, sample in enumerate(val_loader):

                        X_img1, X_img2 = sample

                        if torch.cuda.is_available():
                            X_img1 = X_img1.to(DEVICE)
                            X_img2 = X_img2.to(DEVICE)

                        p1, p2, z1, z2 = self.forward(X_img1, X_img2)

                        loss = (
                            -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
                        )
                        val_loss.append(loss.detach())

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
            return X.cpu()


class SimSiamPlugin(base.PreprocessorPlugin):
    """Classification plugin using predefined Convolutional Neural Networks

    Parameters
    ----------
    conv_net: str,
        Name of the predefined convolutional neural networks
    random_state: int, default 0
        Random seed


    Example:
        >>> from autoprognosis.plugins.prediction import Predictions
        >>> plugin = Predictions(category="preprocessors").get("predefined_cnn", conv_net='AlexNet')
        >>> from sklearn.datasets import load_iris
        >>> # Load data
        >>> plugin.fit_transform(X, y) # returns the probabilities for each class
    """

    def __init__(
        self,
        conv_net: str = "AlexNet",
        nonlin: str = "relu",
        lr: float = 1e-5,
        batch_size: int = 32,
        n_iter: int = 200,
        n_iter_min: int = 10,
        n_iter_print: int = 1,
        patience: int = 5,
        early_stopping: bool = True,
        weight_decay: float = 1e-3,
        output_size: int = 50,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.conv_net = conv_net.lower()
        self.non_lin = nonlin
        self.lr = lr
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.n_iter_min = n_iter_min
        self.patience = patience
        self.early_stopping = early_stopping
        self.n_iter_print = n_iter_print
        self.weight_decay = weight_decay
        self.output_size = output_size

    @staticmethod
    def name() -> str:
        return "simsiam"

    @staticmethod
    def subtype() -> str:
        return "image_reduction"

    @staticmethod
    def modality_type():
        return IMAGE_KEY

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [
            params.Categorical("conv_net", CNN),
            params.Categorical("output_size", [50, 100, 300]),
            params.Integer("lr", 0, 5),
            params.Categorical("data_augmentation", []),
            params.Categorical("weight_decay", [1e-3, 1e-4, 1e-5]),
        ]

    @staticmethod
    def hyperparameter_lr_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [
            params.Categorical("conv_net", CNN),
            params.Categorical("output_size", [50, 100, 300]),
            params.Integer("lr", 0, 5),
            params.Categorical("data_augmentation", []),
            params.Categorical("weight_decay", [1e-3, 1e-4, 1e-5]),
            params.Integer("n_layers", [1, 3]),
        ]

    def sample_hyperparameters(cls, trial, *args: Any, **kwargs: Any):
        param_space = cls.hyperparameter_lr_space(*args, **predefined_args)

        results = {}

        for hp in param_space:
            results[hp.name] = hp.sample(trial)

        return results

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "SimSiamPlugin":

        y = args[0]
        self.n_classes = len(y.value_counts())

        self.model = SimSiam(
            weight_decay=self.weight_decay,
            batch_size=self.batch_size,
            lr=self.lr,
            n_iter_min=self.n_iter_min,
            n_iter=self.n_iter,
            early_stopping=self.early_stopping,
            n_iter_print=self.n_iter_print,
            patience=self.patience,
            base_encoder=CNN_MODEL["resnet34"],
        )

        X_tensor = torch.stack(X.squeeze().tolist())

        self.model.train(X_tensor)

        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        X = torch.stack(X.squeeze().tolist())
        with torch.no_grad():
            results = np.empty(
                (
                    0,
                    self.model.predict(
                        torch.stack((X[0], X[0])).squeeze().to(DEVICE)
                    ).shape[1],
                )
            )
            test_dataset = TensorDataset(X)
            test_loader = DataLoader(
                test_dataset, batch_size=self.batch_size, pin_memory=False
            )
            for batch_test_ndx, X_test in enumerate(test_loader):
                results = np.vstack(
                    (
                        results,
                        self.model.predict(X_test[0].to(DEVICE)).detach().cpu().numpy(),
                    )
                )
            return pd.DataFrame(results)

    def save(self) -> bytes:
        return save_model(self)

    @classmethod
    def load(cls, buff: bytes) -> "SimSiamPlugin":
        return load_model(buff)


plugin = SimSiamPlugin
