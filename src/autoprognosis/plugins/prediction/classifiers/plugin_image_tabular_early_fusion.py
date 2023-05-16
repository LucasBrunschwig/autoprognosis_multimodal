# stdlib
from typing import Any, List

# third party
import numpy as np
import pandas as pd

# autoprognosis absolute
import autoprognosis.plugins.core.params as params
from autoprognosis.plugins.prediction.classifiers import Classifiers
import autoprognosis.plugins.prediction.classifiers.base as base
from autoprognosis.utils.distributions import enable_reproducible_results
from autoprognosis.utils.serialization import load_model, save_model


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
        classifier: str = "logistic_regression",
        random_state: int = 0,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        enable_reproducible_results(random_state)

        # TODO: check how the fact that we have many non-relevant attribute influence the trials
        #       from what I have seen there might be an issue because pruning is only for exact same params

        # Select relevant classifier
        classifier_attribute = {}
        for name, value in kwargs.items():
            if classifier in name:
                attr_name = name[len(classifier) + 1 :]
                classifier_attribute[attr_name] = value

        self.classifier = classifier
        self.model = Classifiers().get_type(classifier)(**classifier_attribute)

        # If we use a pretrained CNN, the image preprocessing is handled by the CNN

        if model is not None:
            self.model = model
            return

    @staticmethod
    def name() -> str:
        return "image_tabular_early_fusion"

    @classmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:

        default_early_fusion_classifier = [
            "random_forest",
            "xgboost",
            "catboost",
            "lgbm",
            "logistic_regression",
            "linear_svm",
        ]

        hp_classifier = []
        for classifier in default_early_fusion_classifier:
            params_classifier = (
                Classifiers().get_type(classifier).hyperparameter_space()
            )
            for param in params_classifier:
                param.name = classifier + "_" + param.name
            hp_classifier.extend(params_classifier)

        return [
            params.Categorical("ratio", [0.3, 0.5, 0.7]),
            params.Categorical("classifier", default_early_fusion_classifier),
        ] + hp_classifier

    def _fit(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> "ImageTabularEarlyFusionPlugin":

        X_images = kwargs["img"]
        kwargs.pop("img")

        # Step 1: Concatenate into a unique vectors
        X = pd.DataFrame(np.concatenate((X.to_numpy(), X_images.to_numpy()), axis=1))

        # Step 2: fit the newly obtained vector with the selected classifier
        self.model.fit(X, *args, **kwargs)

        return self

    def _predict(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        X_images = kwargs["img"]
        kwargs.pop("img")
        X = pd.DataFrame(np.concatenate((X.to_numpy(), X_images.to_numpy()), axis=1))
        return self.model.predict(X, *args, **kwargs)

    def _predict_proba(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> pd.DataFrame:
        X_images = kwargs["img"]
        kwargs.pop("img")
        X = pd.DataFrame(np.concatenate((X.to_numpy(), X_images.to_numpy()), axis=1))
        return self.model.predict_proba(X, *args, **kwargs)

    def save(self) -> bytes:
        return save_model(self.model)

    @classmethod
    def load(cls, buff: bytes) -> "ImageTabularEarlyFusionPlugin":
        model = load_model(buff)

        return cls(model=model)


plugin = ImageTabularEarlyFusionPlugin
