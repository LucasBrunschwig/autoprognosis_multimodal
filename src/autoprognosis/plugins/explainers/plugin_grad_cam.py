# stdlib
from typing import Any, List, Optional

# third party
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# autoprognosis absolute
from autoprognosis.plugins.explainers.base import ExplainerPlugin
from autoprognosis.utils.default_modalities import IMAGE_KEY
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


def get_last_conv_layer_before_classifier(model, input_size, model_name):

    model.eval()  # Set the model to evaluation mode
    model.cpu()  # Set the model to cpu if it was trained on cuda

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

    if model_name in ["alexnet"]:
        # Find the last convolutional layer before the classifier
        last_conv_layer = list(activations.keys())[-1]
    elif model_name in ["resnet50"]:
        last_conv_layer = list(activations.keys())[-2]

    return last_conv_layer


class GradCAM:
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

    def generate_cam_plusplus(self, input_image, target_class, target_layer):
        self.target_layer = target_layer
        self.register_hooks()
        self.model.zero_grad()

        output = self.classifier.predict_proba_tensor(input_image)
        target = output[:, target_class]
        target.backward()

        gradients = self.gradient[0]  # Get gradients
        gradients = gradients.detach().cpu().numpy()
        activations = self.activations  # Get activations
        activations = activations.detach().cpu().numpy()

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
        w = input_image.values[0, 0].size[0]
        h = input_image.values[0, 0].size[1]
        cam = cv2.resize(cam, (w, h))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)

        return cam

    def generate_cam(self, input_image, target_class, target_layer):

        self.target_layer = target_layer
        self.register_hooks()
        self.model.zero_grad()

        output = self.classifier.predict_proba_tensor(input_image)
        target = output[:, target_class]
        target.backward()

        # Compute the gradient for each activation maps k and target class c d(y^c)/dA^k
        gradients = self.gradient[0]
        activations = self.activations[0]

        weights = torch.mean(gradients, dim=(1, 2), keepdim=True)
        cam = torch.sum(weights * activations, dim=0)
        cam = nn.functional.relu(cam)

        cam = cam.detach().cpu().numpy()
        cam = cv2.resize(
            cam, (input_image.squeeze()._size[0], input_image.squeeze()._size[1])
        )
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))

        return cam


class GradCAMPlugin(ExplainerPlugin):
    """
    Interpretability plugin based on grad-CAM

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

    def explain(
        self,
        X: pd.DataFrame,
        label: pd.DataFrame,
        n_top=1,
        target_layer: str = None,
        best: bool = True,
    ) -> dict:

        if target_layer is None:
            # by default grad-cam will extract the last convolutional layer
            input_size = self.classifier.get_size()
            n_channel = 3
            input_size = (n_channel, input_size, input_size)
            target_layer = get_last_conv_layer_before_classifier(
                self.classifier.get_image_model(), input_size, self.classifier.conv_net
            )

        label = pd.DataFrame(LabelEncoder().fit_transform(label))

        results = {label_: [] for label_ in label.squeeze().unique()}

        # Get the top 2 highest score for each class
        predictions = pd.DataFrame(self.estimator.predict_proba(X))
        predictions["label"] = label.squeeze()
        predictions["proba"] = predictions.apply(lambda d: d[d["label"]], axis=1)
        predictions = predictions[["proba", "label"]]
        if best:
            indices = predictions.groupby("label")["proba"].nlargest(n_top)
        else:
            indices = predictions.groupby("label")["proba"].nsmallest(n_top)

        print(indices)
        for label_, ix in indices.index:
            local_X = X.iloc[ix]

            # Preprocess Data
            local_X = pd.DataFrame(local_X)
            for stage in self.estimator.stages[:-1]:
                local_X = stage.transform(local_X)

            # Generate CAM
            cam_normalized = self.explainer.generate_cam_plusplus(
                local_X, label_, target_layer
            )
            cam_normalized = cv2.cvtColor(
                np.uint8(255 * cam_normalized), cv2.COLOR_RGB2BGR
            )
            cam_normalized = cv2.applyColorMap(cam_normalized, cv2.COLORMAP_JET)
            cam_normalized = cv2.cvtColor(cam_normalized, cv2.COLOR_BGR2RGB)

            img_array = np.array(local_X.squeeze())
            alpha = 0.4
            superposed_image = alpha * cam_normalized + (1 - alpha) * img_array
            results[label_].append([img_array / 255.0, superposed_image / 255.0])

        return results

    def plot(
        self,
        X: pd.DataFrame,
        label: pd.DataFrame,
        target_layer: str = None,
        class_names: list = None,
        best: bool = True,
    ) -> dict:

        if target_layer is None:
            # by default grad-cam will extract the last convolutional layer
            input_size = self.estimator.stages[-1].get_size()
            n_channels = 3
            input_size = (n_channels, input_size, input_size)
            target_layer = get_last_conv_layer_before_classifier(
                self.estimator.stages[-1].get_model(),
                input_size,
                self.classifier.conv_net,
            )

        label = pd.DataFrame(LabelEncoder().fit_transform(label))

        results = {label_: [] for label_ in label.squeeze().unique()}

        # Get the top 2 highest score for each class
        predictions = pd.DataFrame(self.estimator.predict_proba(X))
        predictions["label"] = label.squeeze()
        predictions["proba"] = predictions.apply(lambda d: d[d["label"]], axis=1)
        predictions = predictions[["proba", "label"]]

        if best:
            indices = predictions.groupby("label")["proba"].nlargest(1)
        else:
            indices = predictions.groupby("label")["proba"].nsmallest(1)

        for label_, ix in indices.index:
            local_X = X.iloc[ix]

            # Preprocess Data
            local_X = pd.DataFrame(local_X)
            for stage in self.estimator.stages[:-1]:
                local_X = stage.transform(local_X)

            # Generate CAM
            cam_normalized = self.explainer.generate_cam(local_X, label_, target_layer)
            cam_normalized = cv2.applyColorMap(
                np.uint8(255 * cam_normalized), cv2.COLORMAP_JET
            )
            img_array = np.array(local_X.squeeze())
            alpha = 0.4
            superposed_image = alpha * cam_normalized + (1 - alpha) * img_array
            results[label_].append([img_array / 255.0, superposed_image / 255.0])

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

        return results

    def modality_type(self):
        return IMAGE_KEY

    @staticmethod
    def name() -> str:
        return "grad_cam"

    @staticmethod
    def pretty_name() -> str:
        return "Grad_CAM"


plugin = GradCAMPlugin
