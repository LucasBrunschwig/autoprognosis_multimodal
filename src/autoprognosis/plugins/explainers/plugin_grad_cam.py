# stdlib
import copy
from typing import Any, List, Optional

# third party
import cv2
import numpy as np
import pandas as pd

# autoprognosis absolute
from autoprognosis.plugins.explainers.base import ExplainerPlugin
from autoprognosis.utils.pip import install

for retry in range(2):
    try:
        # third party
        import torch
        import torch.nn as nn

        break
    except ImportError:
        depends = ["torch"]
        install(depends)


class GradCAM:
    def __init__(
        self,
        estimator: Any,
        target_layer: str,
    ):

        # if estimator.is_fitted:
        self.model = estimator.stages[-1].model.model
        self.target_layer = target_layer

        self.gradient = None
        self.activations = None
        self.handles = None

        self.register_hooks()

    def register_hooks(
        self,
    ):
        def backward_hook(module, grad_input, grad_output):
            self.gradient = grad_output[0]

        def forward_hook(module, input, output_):
            self.activations = output_

        target_layer = self.target_layer
        target_module = self.model._modules.get(target_layer)

        # Retain the backpropagation gradients of the selected layer
        backward_handle = target_module.register_backward_hook(backward_hook)

        # Retain the activation maps of the selected layers
        forward_handle = target_module.register_forward_hook(forward_hook)

        self.handles = [backward_handle, forward_handle]

    def generate_cam(self, input_image, target_class):
        self.model.zero_grad()
        output = self.model(input_image)
        target = output[:, target_class]

        target.backward()

        # Compute the gradient for each activation maps k and target class c d(y^c)/dA^k
        gradients = self.gradient[0]
        activations = self.activations[0]

        weights = torch.mean(gradients, dim=(1, 2), keepdim=True)
        cam = torch.sum(weights * activations, dim=0)
        cam = nn.functional.relu(cam)

        cam = cam.detach().cpu().numpy()
        cam = cv2.resize(cam, (input_image.shape[3], input_image.shape[2]))
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))

        return cam


class GradCAMPlugin(ExplainerPlugin):
    """
    Interpretability plugin based on LIME.

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
        >>> model = Classifiers().get("cnn")
        >>>
        >>> explainer = Explainers().get(
        >>>     "grad_cam",
        >>>     model,
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
        target_layer: str,
        images: Optional[List] = None,
        task_type: str = "classification",
        prefit: bool = False,
        n_epoch: int = 10000,
        # Risk estimation
        time_to_event: Optional[pd.DataFrame] = None,  # for survival analysis
        eval_times: Optional[List] = None,  # for survival analysis
        random_state: int = 0,
        **kwargs: Any,
    ) -> None:
        if task_type not in ["classification", "risk_estimation"]:
            raise RuntimeError("invalid task type")

        self.task_type = task_type
        self.feature_names = list(
            images if images is not None else pd.DataFrame(X).columns
        )
        super().__init__(self.feature_names)

        self.estimator = copy.deepcopy(estimator)
        if task_type == "classification":
            if not prefit:
                self.estimator.fit(X, y)

        elif task_type == "risk_estimation":
            if time_to_event is None or eval_times is None:
                raise RuntimeError("invalid input for risk estimation interpretability")

        if task_type == "classification":
            self.explainer = GradCAM(estimator, target_layer=target_layer)
        else:
            raise ValueError("Not Implemented")

    def explain(self, X: pd.DataFrame, label: pd.DataFrame) -> pd.DataFrame:
        for (_, img_), (_, label_) in zip(X.iterrows(), label.iterrows()):
            outputs = self.estimator.predict_proba(pd.DataFrame(img_))
        return outputs
        # cam = self.explainer.generate_cam()

    @staticmethod
    def name() -> str:
        return "grad_cam"

    @staticmethod
    def pretty_name() -> str:
        return "Grad_CAM"


plugin = GradCAMPlugin
