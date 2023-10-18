# stdlib
import time
from typing import Any, List, Optional, Union

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
import autoprognosis.logger as log
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.prediction.classifiers.base as base
from autoprognosis.plugins.utils.custom_dataset import (
    TestImageDataset,
    TrainingImageDataset,
    build_data_augmentation_strategy,
    data_augmentation_strategies,
)
from autoprognosis.utils.default_modalities import IMAGE_KEY
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

EPS = 1e-8

NONLIN = {
    "elu": nn.ELU,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "selu": nn.SELU,
}
Learning_Rates = {
    0: [1e-4, 1e-5],
    1: [1e-4, 1e-4],
    2: [1e-5, 1e-6],
    3: [1e-5, 1e-5],
    4: [1e-6, 1e-6],
    5: [1e-6, 1e-7],
}


class ConvNetPredefinedFineTune(nn.Module):
    """Convolutional Neural Network model for early fusion models.

    Parameters
    ----------
    model_name (str):
        model name of one of the default Deep CNN architectures.
    n_classes (int):
        the number of predicted classes
    non_linear (str):
        the non-linearity in the additional layers
    transformation (callable):
        data augmentation strategy applied on-the-fly to images during training
    n_additional_layer (int):
        the number of added layer to the predefined CNN for transfer learning
    non_linear (str):
        the non-linearity of the added layers
    batch_size (int):
        batch size for each step during training
    lr (float):
        learning rate for training, usually lower than the initial training
    n_iter (int):
        the number of iteration
    weight_decay (float):
        l2 (ridge) penalty for the weights.
    early_stopping (bool):
        stopping when the metric did not improve for multiple iterations (max = patience)
    n_iter_print (int):
        logging an update every n iteration
    n_iter_min (int):
        minimum number of iterations
    patience (int):
        the number of iterations before stopping the training with early stopping
    n_additional_layers (int):
        number of additional layers on top of the network
    clipping_value (int):
        clipping parameters value during training
    latent_representation (int): Optional
        size of the latent representation during early fusion
    weighted_cross_entropy (bool):
        use weighted cross entropy during training
    replace_classifier (bool):
        replace the classifier instead of adding layers on top of the classifiers

    """

    def __init__(
        self,
        conv_name: str,
        n_classes: Optional[int],
        non_linear: str,
        transformation: transforms.Compose,
        batch_size: int,
        lr: list,
        n_iter: int,
        weight_decay: float,
        early_stopping: bool,
        n_iter_print: int,
        n_iter_min: int,
        patience: int,
        preprocess,
        n_unfrozen_layer: int = 0,
        n_additional_layers: int = 2,
        clipping_value: int = 1,
        latent_representation: int = None,
        weighted_cross_entropy: bool = False,
        replace_classifier: bool = False,
    ):

        super(ConvNetPredefinedFineTune, self).__init__()

        # Training Parameters
        self.batch_size = batch_size
        self.lr = lr
        self.n_iter = n_iter
        self.n_iter_min = n_iter_min
        self.early_stopping = early_stopping
        self.n_iter_print = n_iter_print
        self.patience = patience
        self.preprocess = preprocess
        self.transforms = transformation
        self.clipping_value = clipping_value
        self.weighted_cross_entropy = weighted_cross_entropy

        # Model Architectures
        self.model_name = conv_name.lower()
        self.latent_representation = (
            latent_representation  # latent representation in early fusion
        )
        self.n_additional_layers = n_additional_layers
        self.model = CNN_MODEL[self.model_name](weights=WEIGHTS[self.model_name])

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze layers in the classifier
        if n_unfrozen_layer > 0 and not replace_classifier:
            n_unfrozen_layer = self.unfreeze_last_n_layers_classifier(n_unfrozen_layer)

        # Unfreeze remaining layers in the convolutional layers
        if n_unfrozen_layer > 0:
            self.unfreeze_last_n_layers_convolutional(n_unfrozen_layer)

        if n_classes is None:
            raise RuntimeError(
                "To build the model, the CNN requires to know the number of classes"
            )

        # Number of features inputs in additional layers
        n_features_in = self.extract_n_features_in(replace_classifier)

        # Define the set of additional layers
        additional_layers = []
        NL = NONLIN[non_linear]

        if n_additional_layers > 0:

            n_intermediate = n_features_in // 2

            additional_layers = [
                nn.Linear(n_features_in, n_intermediate),
                NL(inplace=True),
                nn.Dropout(p=0.5, inplace=False),
            ]

            for i in range(n_additional_layers - 1):
                additional_layers.extend(
                    [
                        nn.Linear(n_intermediate, int(n_intermediate / 2)),
                        NL(inplace=True),
                        nn.Dropout(p=0.5, inplace=False),
                    ]
                )
                n_intermediate = n_intermediate // 2

            # In early fusion, the output size is specified
            if self.latent_representation:
                additional_layers.extend(
                    [
                        nn.Linear(n_intermediate, latent_representation),
                        nn.BatchNorm1d(latent_representation),
                    ]
                )
                additional_layers.append(nn.Linear(latent_representation, n_classes))
                additional_layers[-3].bias.requires_grad = False
            else:
                additional_layers.append(nn.Linear(n_intermediate, n_classes))
        else:
            if self.latent_representation:
                additional_layers.extend(
                    [
                        nn.Linear(n_features_in, latent_representation),
                        nn.BatchNorm1d(latent_representation),
                    ]
                )
                additional_layers.append(nn.Linear(latent_representation, n_classes))
            else:
                additional_layers.append(nn.Linear(n_features_in, n_classes))

        # Define the parameters to optimize
        params_ = self.define_parameters_optimization(
            additional_layers, lr, weight_decay, replace_classifier
        )

        self.model.to(DEVICE)
        self.optimizer = torch.optim.Adam(params_)

    def extract_n_features_in(self, replace_classifier: bool = False):
        """Extract the size of the input features to define the additional layers.

        Parameters
        ----------
        replace_classifier (bool): specify if the additional layers replace the classifier.
        """

        def get_in_features_from_classifier(module, replace_classifier_=False):
            """Helper function to find the first or last linear input size of the classifier"""
            n_features_in = None
            if isinstance(module, torch.nn.Sequential):
                for layer in module:
                    if isinstance(layer, torch.nn.modules.linear.Linear):
                        n_features_in = layer.in_features
                        if replace_classifier_:
                            break
            elif isinstance(module, torch.nn.modules.linear.Linear):
                n_features_in = module.in_features
            else:
                raise ValueError(f"Unknown Classifier Architecture {self.model_name}")

            return n_features_in

        if hasattr(self.model, "fc"):
            classifier = self.model.fc
        elif hasattr(self.model, "classifier"):
            classifier = self.model.classifier
        else:
            raise ValueError(f"Unknown Classifier Module Name: {self.model_name}")

        return get_in_features_from_classifier(classifier, replace_classifier)

    def define_parameters_optimization(
        self, additional_layers, lr, weight_decay, replace_classifier
    ):
        params_ = []

        def setup_params(classifier_attr_name):
            """Set up the parameters for optimization"""
            classifier = getattr(self.model, classifier_attr_name)

            if (
                isinstance(classifier, torch.nn.modules.Sequential)
                and not replace_classifier
            ):
                classifier[-1] = nn.Sequential(*additional_layers)
                name_match = f"{classifier_attr_name}.{len(classifier) - 1}"
            else:
                setattr(
                    self.model, classifier_attr_name, nn.Sequential(*additional_layers)
                )
                name_match = classifier_attr_name

            for name, param in self.model.named_parameters():
                if name_match in name:
                    param.requires_grad = True
                    params_.append(
                        {"params": param, "lr": lr[0], "weight_decay": weight_decay}
                    )
                elif param.requires_grad:
                    params_.append(
                        {"params": param, "lr": lr[1], "weight_decay": weight_decay}
                    )

        if hasattr(self.model, "fc"):
            setup_params("fc")
        elif hasattr(self.model, "classifier"):
            setup_params("classifier")

        return params_

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
        if isinstance(self.model, torch.nn.Sequential):
            feature_extractor = self.model
        elif hasattr(self.model, "features"):  # For most architectures
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
        if isinstance(self.model, torch.nn.Sequential):
            feature_extractor = self.model
        elif hasattr(self.model, "classifier"):
            feature_extractor = self.model.classifier
        elif hasattr(self.model, "fc"):  # For ResNet
            feature_extractor = self.model.fc
        else:
            raise ValueError("Unsupported Architecture")

        # Perform depth-first search on the feature extractor
        dfs(feature_extractor)

        return n - unfrozen_count

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_model(self):
        """Returns the model necessary for grad-cam++ computation"""
        return self.model

    def set_zero_grad(self):
        self.model.zero_grad()

    def train_(self, X: pd.DataFrame, y: torch.Tensor) -> "ConvNetPredefinedFineTune":

        y = self._check_tensor(y).squeeze().long()

        dataset = TrainingImageDataset(
            X, y, preprocess=self.preprocess, transform=self.transforms
        )

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
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
        self.model.train()

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
                            log.info(
                                f"Final Epoch: {i}, loss: {val_loss:.4f}, train_loss: {torch.mean(train_loss):.4f}"
                            )
                            break

                    if i % self.n_iter_print == 0:
                        log.trace(
                            f"Epoch: {i}, loss: {val_loss:.4f}, train_loss: {torch.mean(train_loss):.4f}, epoch elapsed time: {(end_ - start_):.2f}"
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
                if isinstance(self.model.classifier[-1], torch.nn.Sequential):
                    self.model.classifier[-1][-1] = nn.Identity()
                else:
                    self.model.classifier[-1] = nn.Identity()

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
    conv_name: str,
        the predefined architecture
    n_additional_layer (int):
        the number of added layer to the predefined CNN for transfer learning
    replace_classifier (bool):
        replace the classifier instead of adding layers on top of the classifiers
    non_linear (str):
        the non-linearity in the additional layers
    data_augmentation (str):
        data augmentation strategy applied on-the-fly to images during training
    lr: float
        learning rate for optimizer. step_size equivalent in the JAX version.
    n_unfrozen_layers:
        the number of layer to unfreeze
    weightec_cross_entropy (bool):
        use weighted cross entropy during training
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
    random_state: int, default 0
        Random seed

    # Example:
         >>> from autoprognosis.plugins.prediction import Predictions
         >>> plugin = Predictions(category="classifier").get("cnn_fine_tune")
         >>> from sklearn.datasets import load_digits
         >>> from PIL import Image
         >>> import numpy as np
         >>> # load data
         >>> X, y = load_digits(return_X_y=True, as_frame=True)
         >>> # Transform X into PIL Images
         >>> X["image"] = X.apply(lambda row: Image.fromarray(np.stack([(row.to_numpy().reshape((8, 8))).astype(np.uint8)]*3, axis=-1)), axis=1)
         >>> plugin.fit_predict(X[["image"]], y)
    """

    def __init__(
        self,
        # Architecture
        conv_name: str = "alexnet",
        n_additional_layers: int = 2,
        non_linear: str = "relu",
        replace_classifier: bool = False,
        # Data Augmentation
        data_augmentation: Union[str, transforms.Compose] = "",
        # Training
        lr: int = 3,
        n_unfrozen_layers: int = 2,
        weighted_cross_entropy: bool = False,
        weight_decay: float = 1e-4,
        n_iter: int = 1000,
        batch_size: int = 128,
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

        # CNN Architecture
        self.conv_name = conv_name
        self.replace_classifier = replace_classifier
        self.non_linear = non_linear
        self.n_classes = None  # Defined during training
        self.n_additional_layers = n_additional_layers

        # Training Parameters
        self.lr = Learning_Rates[lr]
        self.n_unfrozen_layer = n_unfrozen_layers
        self.weighted_cross_entropy = weighted_cross_entropy
        self.weight_decay = weight_decay
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.n_iter_print = n_iter_print
        self.patience = patience
        self.n_iter_min = n_iter_min
        self.early_stopping = early_stopping
        self.clipping_value = clipping_value

        # Data Augmentation
        self.preprocess = self.image_preprocess()
        self.data_augmentation = build_data_augmentation_strategy(data_augmentation)

        # Ensure baseline is consistent with selected architecture
        if (
            predefined_args.get("predefined_cnn", None)
            and len(predefined_args["predefined_cnn"]) > 0
        ):
            self.conv_name = predefined_args["predefined_cnn"][0]

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
            params.Categorical("conv_name", CNN),
            params.Integer("n_additional_layers", 0, 3),
            params.Categorical("replace_classifier", [True, False]),
            # Training
            params.Integer("lr", 0, 5),
            params.Integer("n_unfrozen_layer", 0, 5),
            params.Categorical("weighted_cross_entropy", [True, False]),
            params.Categorical("clipping_value", [0, 1]),
            # Data Augmentation
            params.Categorical("data_augmentation", data_augmentation_strategies),
        ]

    def image_preprocess(self):
        weights = models.get_weight(WEIGHTS[self.conv_name.lower()])
        return weights.transforms(antialias=True)

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
            conv_name=self.conv_name,
            n_classes=n_classes,
            non_linear=self.non_linear,
            transformation=self.data_augmentation,
            batch_size=self.batch_size,
            lr=self.lr,
            n_iter=self.n_iter,
            weight_decay=self.weight_decay,
            early_stopping=self.early_stopping,
            n_iter_print=self.n_iter_print,
            n_iter_min=self.n_iter_min,
            patience=self.patience,
            preprocess=self.preprocess,
            n_unfrozen_layer=self.n_unfrozen_layer,
            n_additional_layers=self.n_additional_layers,
            clipping_value=self.clipping_value,
            weighted_cross_entropy=self.weighted_cross_entropy,
            replace_classifier=self.replace_classifier,
        )

        self.model.train_(X, y)

        return self

    def _predict(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        self.model.to(DEVICE)
        self.model.eval()
        with torch.no_grad():
            results = np.empty((0, 1))
            test_loader = DataLoader(
                TestImageDataset(X, preprocess=self.preprocess),
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
        """This method forces model to CPU with gradients for grad-CAM++"""
        # TMP LUCAS: check if necessary
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        self.model.cpu()
        self.model.eval()
        results = torch.empty((0, self.n_classes))
        test_dataset = TestImageDataset(X, preprocess=self.preprocess)
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
        self.model.eval()
        with torch.no_grad():
            results = np.empty((0, self.n_classes))
            test_dataset = TestImageDataset(X, preprocess=self.preprocess)
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
            return pd.DataFrame(results)

    def set_zero_grad(self):
        self.model.zero_grad()

    def get_image_model(self):
        return self.model.get_model()

    def get_size(self):
        return models.get_weight(WEIGHTS[self.conv_name]).transforms.keywords[
            "crop_size"
        ]

    def get_conv_name(self):
        return self.conv_name

    def eval(self):
        self.model.eval()

    def save(self) -> bytes:
        return save_model(self)

    @classmethod
    def load(cls, buff: bytes) -> "CNNFineTunePlugin":
        return load_model(buff)


plugin = CNNFineTunePlugin
