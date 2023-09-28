# stdlib
from typing import Any, List, Optional

# third party
from captum.attr import IntegratedGradients
import numpy as np
import pandas as pd
from torchvision import transforms

# autoprognosis absolute
from autoprognosis.plugins.explainers.base import ExplainerPlugin
from autoprognosis.utils.default_modalities import (
    IMAGE_KEY,
    MULTIMODAL_KEY,
    TABULAR_KEY,
)
from autoprognosis.utils.pip import install

for retry in range(2):
    try:
        # third party
        import torch

        break
    except ImportError:
        depends = ["torch"]
        install(depends)


class IntegratedGradientPlugin(ExplainerPlugin):
    """
    Interpretability plugin based on Integrated Gradients Method

    Args:
        estimator: model. The model to explain.
        X: dataframe. Training set
        y: dataframe. Training labels
        task_type: str. classification of risk_estimation
        prefit: bool. If true, the estimator won't be trained.
        n_epoch: int. training epochs
        time_to_event: dataframe. Used for risk estimation tasks.
        eval_times: list. Used for risk estimation tasks.

    Example:
        >>> import pandas as pd
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.model_selection import train_test_split
        >>>from autoprognosis.plugins.explainers import Explainers
        >>> from autoprognosis.plugins.prediction.classifiers import Classifiers
        >>>
        >>> X, y = load_iris(return_X_y=True)
        >>>
        >>> X = pd.DataFrame(X)
        >>> y = pd.Series(y)
        >>>
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        >>> model = Classifiers(category="multimodal").get("intermediate_conv_net")
        >>>
        >>> explainer = Explainers().get(
        >>>     "grad_cam",
        >>>     model,
        >>>     X_train,
        >>>     y_train,
        >>>     task_type="classification",
        >>> )
        >>>
        >>> explainer.explain(X_test)
    """

    def __init__(
        self,
        estimator: Any,
        X: pd.DataFrame,
        y: pd.DataFrame,
        baselines: tuple,
        images: Optional[List] = None,
        task_type: str = "classification",
        prefit: bool = False,
        # Risk estimation
        time_to_event: Optional[pd.DataFrame] = None,  # for survival analysis
        eval_times: Optional[List] = None,  # for survival analysis
        **kwargs: Any,
    ) -> None:
        if task_type not in ["classification", "risk_estimation"]:
            raise RuntimeError("invalid task type")

        self.task_type = task_type
        self.feature_names = list(
            images if images is not None else pd.DataFrame(X[IMAGE_KEY]).columns
        )
        self.feature_names = self.feature_names + pd.DataFrame(X[TABULAR_KEY]).columns
        super().__init__(self.feature_names)

        self.estimator = estimator

        self.X_img = X[IMAGE_KEY]
        self.X_tab = X[TABULAR_KEY]

        if task_type == "classification":
            if not prefit:
                self.estimator.fit(X, y)

        elif task_type == "risk_estimation":
            if time_to_event is None or eval_times is None:
                raise RuntimeError("invalid input for risk estimation interpretability")

        if task_type == "classification":
            # Extract the model to cpu
            self.explainer = IntegratedGradients(
                self.estimator.get_classifier().model.cpu()
            )
            self.baselines = tuple(baselines)

        else:
            raise ValueError("Not Implemented")

    def explain(self, X: dict, y: pd.DataFrame) -> dict:
        results = [[], []]

        y = torch.from_numpy(np.asarray(y))

        X_img = X[IMAGE_KEY].copy()
        X_tab = X[TABULAR_KEY].copy()

        tab_baselines = self.baselines[0]
        for stage in self.estimator.stages[:-1]:
            tab_baselines = pd.DataFrame(tab_baselines)
            tab_baselines = stage.transform(tab_baselines)
            X_tab = stage.transform(X_tab)
        baseline_transformed = tuple(
            (torch.from_numpy(np.asarray(tab_baselines)), self.baselines[1])
        )
        for i in range(len(X[TABULAR_KEY])):
            img = transforms.ToTensor()(X_img.iloc[0][0])
            img = self.estimator.get_classifier().preprocess(img)
            img.requires_grad = True
            tab = torch.from_numpy(np.asarray(X_tab.iloc[0]))
            tab.requires_grad = True
            attr_tab, attr_img = self.explainer.attribute(
                (tab.unsqueeze(dim=0).cpu(), img.unsqueeze(dim=0).cpu()),
                baselines=baseline_transformed,
                target=y[i],
                n_steps=1000,
            )

            results[0].append(attr_tab)
            results[1].append(attr_img)

        return results

    def plot(
        self,
        X: pd.DataFrame,
    ) -> dict:

        pass

    def modality_type(self):
        return MULTIMODAL_KEY

    @staticmethod
    def name() -> str:
        return "integrated_gradients"

    @staticmethod
    def pretty_name() -> str:
        return "Integrated_Gradients"


plugin = IntegratedGradientPlugin