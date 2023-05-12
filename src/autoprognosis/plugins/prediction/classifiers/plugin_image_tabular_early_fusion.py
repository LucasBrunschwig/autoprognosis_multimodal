# stdlib
from typing import Any, List

# third party
import numpy as np
import pandas as pd

# autoprognosis absolute
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.prediction.classifiers.base as base
from autoprognosis.plugins.prediction.classifiers.plugin_logistic_regression import (
    LogisticRegressionPlugin,
)
from autoprognosis.utils.distributions import enable_reproducible_results
from autoprognosis.utils.serialization import load_model, save_model

CLASSIFIER = {"LOGREG": LogisticRegressionPlugin}


class ImageTabularEarlyFusionPlugin(base.ClassifierPlugin):
    """Classification plugin based on an early fusion for medical images and clinical data

    Method:

    Args:
        criterion: int
            The function to measure the quality of a split. Supported criteria are “gini”(0) for the Gini impurity and “entropy”(1) for the information gain.

    Example:
        >>> from autoprognosis.plugins.prediction import Predictions
        >>> plugin = Predictions(category="classifiers").get("image_tabular_early_fusion")
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> plugin.fit_predict(X, y) # returns the probabilities for each class
    """

    def __init__(
        self,
        model: Any = None,
        image_feature_reduction: str = "CNN",
        classifier: str = "LOGREG",
        ratio: float = 0.5,
        random_state: int = 0,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        enable_reproducible_results(random_state)

        self.classifier = classifier
        self.image_model_name = image_feature_reduction
        self.ratio = ratio
        self.model = CLASSIFIER[self.classifier.upper()](**kwargs)

        # If we use a pretrained CNN, the image preprocessing is handled by the CNN
        if image_feature_reduction == "CNN":
            self.use_pretrained = True

        if model is not None:
            self.model = model
            return

    @staticmethod
    def name() -> str:
        return "image_tabular_early_fusion"

    @classmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [
            params.Categorical("ratio", [0.3, 0.5, 0.7]),
            params.Categorical("tabular_processing", ["nop"]),
            params.Categorical("classifier", ["LOGREG"]),
        ] + LogisticRegressionPlugin.hyperparameter_space()

    def _fit(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> "ImageTabularEarlyFusionPlugin":

        X_images = kwargs["img"].squeeze()

        # Step 1: Prepare images for concatenation

        # n_tabular = X.shape[1]
        # n_image = int(n_tabular / self.ratio)

        # # if self.image_model_name == "CNN":
        #     X_images = self.image_model.fit_transform(X_images)
        #
        # elif self.image_model_name in ["PCA", "ICA"]:
        #     self.image_model.n_components = n_image
        #     X_images = X_images.apply(lambda img_: transforms.ToTensor()(img_).flatten().cpu().detach().numpy())
        #     X_images = pd.DataFrame(np.stack(X_images.to_numpy().squeeze()))
        #     X_images = self.image_model.fit_transform(X_images)
        #
        # elif self.image_model_name == "AE":
        #     # Needs to train an AE or vAE or de-noising AE to compress data to a latent space
        #     pass

        # Step 2: Concatenate into a unique vectors
        X = pd.DataFrame(np.concatenate((X.to_numpy(), X_images.to_numpy()), axis=1))

        # Step 3: fit the newly obtained vector with the selected classifier
        self.model.fit(X, *args, **kwargs)

        return self

    def _predict(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return self.model.predict(X, *args, **kwargs)

    def _predict_proba(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> pd.DataFrame:
        return self.model.predict_proba(X, *args, **kwargs)

    def save(self) -> bytes:
        return save_model(self.model)

    @classmethod
    def load(cls, buff: bytes) -> "ImageTabularEarlyFusionPlugin":
        model = load_model(buff)

        return cls(model=model)


plugin = ImageTabularEarlyFusionPlugin
