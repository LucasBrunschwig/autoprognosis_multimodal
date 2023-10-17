# stdlib
from typing import Any, List, Optional, Union

# third party
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# autoprognosis absolute
from autoprognosis.plugins.explainers.base import ExplainerPlugin
from autoprognosis.utils.default_modalities import IMAGE_KEY, TABULAR_KEY
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


def get_last_conv_layer_before_classifier(model, input_size):
    input_size = (3, input_size, input_size)

    features = nn.Sequential(*list(model.children())[:-1]).cpu()
    features.eval()

    for name, layer in reversed(list(features.named_children())):
        print(name, layer)

    # Register hooks to store activations in each layer
    activations = {}

    def hook_fn(module_name):
        def hook(module, input, output):
            activations[module_name] = output

        return hook

    hooks = []
    for name, module in model.named_modules():
        if (
            isinstance(module, nn.Conv2d)
            or isinstance(module, nn.AdaptiveAvgPool2d)
            or isinstance(module, nn.AvgPool2d)
        ):
            hooks.append(module.register_forward_hook(hook_fn(name)))

    # Forward pass to collect activations
    input_tensor = torch.rand(*input_size).unsqueeze(0)
    with torch.no_grad():
        model(input_tensor)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    for layer_name, activation in reversed(list(activations.items())):
        if activation.shape[-1] < 2:
            continue
        return layer_name


class GradCAM:
    """
    Grad-CAM++ class, computes the explainable maps given an image model
    """

    def __init__(
        self,
        classifier: Any,
    ):
        self.classifier = classifier
        self.model = classifier.get_image_model()

        self.gradient = None
        self.activations = None
        self.handles = None
        self.target_layer = None

    def register_hooks(self, target_layer):
        def backward_hook(module, grad_input, grad_output):
            self.gradient = grad_output[0]

        def forward_hook(module, input, output_):
            self.activations = output_

        def get_layer(model, layer_str):
            layers = layer_str.split(".")
            for layer in layers:
                model = getattr(model, layer)
            return model

        target_module = get_layer(self.model, target_layer)

        backward_handle = target_module.register_backward_hook(backward_hook)
        forward_handle = target_module.register_forward_hook(forward_hook)

        self.handles = [backward_handle, forward_handle]

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def generate_cam_plusplus(self, input_image, target_class, target_layer):
        self.register_hooks(target_layer)
        self.model.zero_grad()

        output = self.classifier.predict_proba_tensor(input_image)
        target = output[:, target_class]
        target.backward()

        gradients = self.gradient[0].detach().cpu().numpy()
        activations = self.activations.detach().cpu().numpy()

        grads_power_2 = gradients**2
        grads_power_3 = grads_power_2 * gradients
        # Equation 19 in https://arxiv.org/abs/1710.11063
        sum_activations = np.sum(activations, axis=(2, 3))
        eps = 0.000001
        aij = grads_power_2 / (
            2 * grads_power_2 + sum_activations[:, :, None, None] * grads_power_3 + eps
        )
        # Now bring back the ReLU from eq.7 in the paper,
        # And zero out aijs where the activations are 0
        aij = np.where(gradients != 0, aij, 0)

        weights = np.maximum(gradients, 0) * aij
        weights = np.sum(weights, axis=(2, 3))

        cam = np.sum(weights[0][:, np.newaxis, np.newaxis] * activations[0], axis=0)

        cam = np.maximum(cam, 0)
        if isinstance(input_image, dict):
            w = input_image[IMAGE_KEY].values[0, 0].size[0]
            h = input_image[IMAGE_KEY].values[0, 0].size[1]
        else:
            w = input_image.values[0].size[0]
            h = input_image.values[0].size[1]
        cam = cv2.resize(cam, (w, h))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)

        self.remove_hooks()

        return cam


TARGET_LAYER = {
    "alexnet": "",
    "resnet": "",
}


class GradCAMPlugin(ExplainerPlugin):
    """
    Interpretability plugin based on Grad-CAM++ maps

    Args:
        estimator: model. The model to explain.
        X: dataframe or dict. Training set
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
            images if images is not None else pd.DataFrame(X).columns
        )
        super().__init__(self.feature_names)

        self.estimator = estimator
        self.classifier = estimator.get_classifier()

        if task_type == "classification":
            if not prefit:
                self.estimator.fit(X, y)

        elif task_type == "risk_estimation":
            if time_to_event is None or eval_times is None:
                raise RuntimeError("invalid input for risk estimation interpretability")

        if task_type == "classification":
            self.explainer = GradCAM(self.classifier)
        else:
            raise ValueError("Not Implemented")

    def preprocess_data(self, X: Union[pd.DataFrame, dict]):
        if isinstance(X, dict):
            local_X = {
                IMAGE_KEY: X[IMAGE_KEY].copy(),
                TABULAR_KEY: X[TABULAR_KEY].copy(),
            }
            for stage in self.estimator.stages[:-1]:
                if stage.modality_type() == TABULAR_KEY:
                    local_X[TABULAR_KEY] = stage.transform(local_X[TABULAR_KEY])
                elif stage.modality_type() == IMAGE_KEY:
                    local_X[IMAGE_KEY] = stage.transform(local_X[IMAGE_KEY])
        else:
            local_X = X.copy()
            local_X = pd.DataFrame(local_X)

            # Preprocess Data
            for stage in self.estimator.stages[:-1]:
                local_X = stage.transform(local_X)

        return local_X

    @staticmethod
    def superimpose_images(cam, orig):
        # Superimpose Grad-CAM++ Maps and original image
        cam = cv2.cvtColor(np.uint8(255 * cam), cv2.COLOR_RGB2BGR)
        cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
        if isinstance(orig, dict):
            img_array = np.array(orig[IMAGE_KEY].squeeze())
        else:
            img_array = np.array(orig.squeeze())

        alpha = 0.4
        return alpha * cam + (1 - alpha) * img_array, img_array

    def explain(
        self,
        X: Union[pd.DataFrame, dict],
        label: pd.Series,
        target_layer: Optional[str] = None,
        n_top: Optional[int] = 3,
    ) -> (dict, Any):
        """Explain selected feature maps

        X: pd.DataFrame, dict
            Dataset used for explainable interpretation
        label: pd.Series
            labels
        n_top (optional): int,
            explain the n-top predictions for each class
        target_layer (optional): str,
            layer of the convolutional neural network to explain
        """

        target_layer = get_last_conv_layer_before_classifier(
            self.classifier.get_image_model(), input_size=self.classifier.get_size()
        )

        label = pd.DataFrame(LabelEncoder().fit_transform(label))

        results = {label_: [] for label_ in label.squeeze().unique()}

        # Extract the n-top predictions for each class
        predictions = pd.DataFrame(self.estimator.predict_proba(X))
        predictions["label"] = label.squeeze()
        predictions["proba"] = predictions.apply(lambda d: d[d["label"]], axis=1)
        predictions = predictions[["proba", "label"]]
        indices = predictions.groupby("label")["proba"].nlargest(n_top)

        local_X = self.preprocess_data(X)

        for label_, ix in indices.index:

            if isinstance(local_X, dict):
                local_X_single = {
                    IMAGE_KEY: pd.DataFrame(local_X[IMAGE_KEY].iloc[ix]),
                    TABULAR_KEY: pd.DataFrame(local_X[TABULAR_KEY].iloc[ix]),
                }
            else:
                local_X_single = local_X.iloc[ix]

            cam = self.explainer.generate_cam_plusplus(
                local_X_single, label_, target_layer
            )
            superposed_image, img_array = self.superimpose_images(cam, local_X_single)
            results[label_].append([img_array / 255.0, superposed_image / 255.0])

        return results, indices

    def plot(
        self,
        X: pd.DataFrame,
        label: pd.Series,
        target_layer: Optional[str] = None,
        class_names: Optional[list] = None,
    ) -> dict:

        results = self.explain(X, label, target_layer, n_top=1)

        fig, axes = plt.subplots(2, len(results), figsize=(6 * len(results), 12))
        for label, images in results.items():
            image = images[0]
            if class_names is not None:
                axes[0, label].set_title(class_names[label], fontsize=17)

            axes[0, label].imshow(image[0])
            axes[0, label].axis("off")

            axes[1, label].imshow(image[1])
            axes[1, label].axis("off")

        axes[0, 0].get_yaxis().set_visible(False)
        axes[1, 0].get_yaxis().set_visible(False)

        # Add row names to the second row
        fig.text(0.06, 0.7, "Original", ha="center", va="center", fontsize=17)
        fig.text(0.06, 0.27, "Grad-CAM", ha="center", va="center", fontsize=17)

        plt.subplots_adjust(wspace=0.3, hspace=0.3)

    def modality_type(self):
        return IMAGE_KEY

    @staticmethod
    def name() -> str:
        return "grad_cam"

    @staticmethod
    def pretty_name() -> str:
        return "Grad_CAM"


plugin = GradCAMPlugin
