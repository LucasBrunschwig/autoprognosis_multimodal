# stdlib
from typing import Any, Callable, Dict, Tuple, Type, Union

# third party
import numpy as np
from optuna.trial import Trial
import pandas as pd

# autoprognosis absolute
import autoprognosis.plugins.utils.decorators as decorators
from autoprognosis.utils.default_modalities import IMAGE_KEY, TABULAR_KEY
import autoprognosis.utils.serialization as serialization


def _generate_name_impl(plugins: Tuple[Type, ...]) -> Callable:
    def name_impl(*args: Any) -> str:
        return "->".join(p.name() for p in plugins)

    return name_impl


def _generate_type_impl(plugins: Tuple[Type, ...]) -> Callable:
    def type_impl(*args: Any) -> str:
        return "->".join(p.type() for p in plugins)

    return type_impl


def _generate_hyperparameter_space_impl(plugins: Tuple[Type, ...]) -> Callable:
    def hyperparameter_space_impl(*args: Any, **kwargs: Any) -> Dict:
        out = {}
        for p in plugins:
            out[p.name()] = p.hyperparameter_space(*args, **kwargs)
        return out

    return hyperparameter_space_impl


def _generate_hyperparameter_space_for_layer_impl(
    plugins: Tuple[Type, ...]
) -> Callable:
    def hyperparameter_space_for_layer_impl(
        layer: str, *args: Any, **kwargs: Any
    ) -> Dict:
        for p in plugins:
            if p.name() == layer:
                return p.hyperparameter_space(*args, **kwargs)
        raise ValueError(f"invalid layer {layer}")

    return hyperparameter_space_for_layer_impl


def _generate_sample_param_impl(plugins: Tuple[Type, ...]) -> Callable:
    def sample_param_impl(trial: Trial, *args: Any, **kwargs: Any) -> Dict:
        sample: dict = {}
        for p in plugins:
            sample[p.name()] = p.sample_hyperparameters(trial)

        return sample

    return sample_param_impl


def _generate_constructor() -> Callable:
    def _sanity_checks(plugins: Tuple[Type, ...]) -> None:
        if len(plugins) == 0:
            raise RuntimeError("invalid empty pipeline.")

        predictors = 0
        for pl in plugins:
            if pl.type() == "prediction":
                predictors += 1

        if predictors != 1 or plugins[-1].type() != "prediction":
            raise RuntimeError("The last plugin of a pipeling must be a predictor")

    def init_impl(self: Any, args: dict = {}, output: str = "pandas") -> None:
        _sanity_checks(self.plugin_types)

        self.stages = []
        self.args = args

        if output not in ["pandas", "numpy"]:
            raise RuntimeError("Invalid output type")
        if output == "pandas":
            self.output = pd.DataFrame
        elif output == "numpy":
            self.output = np.asarray

        for plugin_type in self.plugin_types:
            plugin_args = {}
            if plugin_type.name() in args:
                plugin_args = args[plugin_type.name()]
            self.stages.append(plugin_type(**plugin_args))

    return init_impl


def _generate_change_output() -> Callable:
    def change_output_impl(self: Any, output: str) -> None:
        if output not in ["pandas", "numpy"]:
            raise RuntimeError("Invalid output type")
        if output == "pandas":
            self.output = pd.DataFrame
        elif output == "numpy":
            self.output = np.asarray

    return change_output_impl


def _generate_get_args() -> Callable:
    def get_args_impl(self: Any) -> Dict:
        return self.args

    return get_args_impl


def _generate_fit(multimodal_type: str) -> Callable:

    if multimodal_type is None:

        def fit_impl(self: Any, X: pd.DataFrame, *args: Any, **kwargs: Any) -> Any:

            local_X = X.copy()

            for stage in self.stages[:-1]:
                local_X = pd.DataFrame(local_X)
                local_X = stage.fit_transform(local_X)

            self.stages[-1].fit(local_X, *args, **kwargs)

            return self

    elif multimodal_type == "late_fusion":

        def fit_impl(
            self: Any, X: Union[dict, pd.DataFrame], *args: Any, **kwargs: Any
        ) -> Any:

            if isinstance(X, dict):
                modality = self.stages[-1].modality_type()
                local_X = X[modality].copy()
            else:
                local_X = X.copy()

            for stage in self.stages[:-1]:
                local_X = pd.DataFrame(local_X)
                local_X = stage.fit_transform(local_X)

            self.stages[-1].fit(local_X, *args, **kwargs)

            return self

    elif multimodal_type == "early_fusion":

        def fit_impl(self: Any, X: dict, *args: Any, **kwargs: Any) -> Any:

            local_X_tab = X[TABULAR_KEY].copy()
            local_X_img = X[IMAGE_KEY].copy()

            # Transform the image
            if X.get("lr", None) is None:
                for stage in self.stages[:-2]:
                    local_X_img = pd.DataFrame(local_X_img)
                    if stage.modality_type() == IMAGE_KEY and not local_X_img.empty:
                        local_X_img = stage.fit_transform(local_X_img, *args)
            else:
                local_X_img = X["lr"]

            # Process tabular
            for stage in self.stages[:-2]:
                local_X_tab = pd.DataFrame(local_X_tab)

                if stage.modality_type() == TABULAR_KEY and not local_X_tab.empty:
                    local_X_tab = stage.fit_transform(local_X_tab)

            local_X = {TABULAR_KEY: local_X_tab, IMAGE_KEY: local_X_img}

            # Modalities Fusion
            local_X = self.stages[-2].fit_transform(local_X)

            # Fit the classifier
            self.stages[-1].fit(local_X, *args, **kwargs)

            return self

    elif multimodal_type == "intermediate_fusion":

        def fit_impl(self: Any, X: dict, *args: Any, **kwargs: Any) -> Any:

            local_X_tab = X[TABULAR_KEY].copy()
            local_X_img = X[IMAGE_KEY].copy()

            for stage in self.stages[:-1]:
                local_X_tab = pd.DataFrame(local_X_tab)
                local_X_img = pd.DataFrame(local_X_img)
                if stage.modality_type() == TABULAR_KEY and not local_X_tab.empty:
                    local_X_tab = stage.fit_transform(local_X_tab)
                elif stage.modality_type() == IMAGE_KEY and not local_X_img.empty:
                    local_X_img = stage.fit_transform(
                        local_X_img, *args, **{"n_tab": local_X_tab.shape[1]}
                    )

            # combine data
            local_X = {"tab": local_X_tab, "img": local_X_img}

            # Fit the classifier
            self.stages[-1].fit(local_X, *args, **kwargs)

            return self

    else:
        raise ValueError(f"Multimodal Type Unknown {multimodal_type}")

    return fit_impl


def _generate_is_fitted() -> Callable:
    def fit_impl(self: Any) -> Any:
        return self.stages[-1].is_fitted()

    return fit_impl


def _generate_predict(multimodal_type) -> Callable:

    if multimodal_type is None:

        @decorators.benchmark
        def predict_impl(
            self: Any, X: pd.DataFrame, *args: Any, **kwargs: Any
        ) -> pd.DataFrame:

            local_X = X.copy()

            for stage in self.stages[:-1]:
                local_X = pd.DataFrame(local_X)
                local_X = stage.transform(local_X)

            result = self.stages[-1].predict(local_X, *args, **kwargs)

            return self.output(result)

    elif multimodal_type == "late_fusion":

        @decorators.benchmark
        def predict_impl(self: Any, X: dict, *args: Any, **kwargs: Any) -> pd.DataFrame:

            if isinstance(X, dict):
                modality = self.stages[-1].modality_type()
                local_X = X[modality].copy()
            else:
                local_X = X.copy()

            for stage in self.stages[:-1]:
                local_X = pd.DataFrame(local_X)
                local_X = stage.transform(local_X)

            result = self.stages[-1].predict(local_X, *args, **kwargs)

            return self.output(result)

    elif multimodal_type == "early_fusion":

        @decorators.benchmark
        def predict_impl(self: Any, X: dict, *args: Any, **kwargs: Any) -> Any:

            local_X_tab = X[TABULAR_KEY].copy()
            local_X_img = X[IMAGE_KEY].copy()

            if X.get("lr", None) is None:
                # Process images
                for stage in self.stages[:-2]:
                    local_X_img = pd.DataFrame(local_X_img)
                    if stage.modality_type() == IMAGE_KEY and not local_X_img.empty:
                        local_X_img = stage.transform(local_X_img, *args)
            else:
                local_X_img = X["lr"]

            # Process tabular
            for stage in self.stages[:-2]:
                local_X_tab = pd.DataFrame(local_X_tab)
                if stage.modality_type() == TABULAR_KEY and not local_X_tab.empty:
                    local_X_tab = stage.transform(local_X_tab)

            local_X = {IMAGE_KEY: local_X_tab, TABULAR_KEY: local_X_img}

            # Fusion plugin
            local_X = self.stages[-2].transform(local_X)

            # Fit the model
            results = self.stages[-1].predict(local_X, *args, **kwargs)

            return self.output(results)

    elif multimodal_type == "intermediate_fusion":

        @decorators.benchmark
        def predict_impl(self: Any, X: dict, *args: Any, **kwargs: Any) -> Any:

            local_X_tab = X[TABULAR_KEY].copy()
            local_X_img = X[IMAGE_KEY].copy()

            # Process Image and Tabular separately
            for stage in self.stages[:-1]:
                local_X_tab = pd.DataFrame(local_X_tab)
                local_X_img = pd.DataFrame(local_X_img)
                if stage.modality_type() == TABULAR_KEY and not local_X_tab.empty:
                    local_X_tab = stage.transform(local_X_tab)
                elif stage.modality_type() == IMAGE_KEY and not local_X_img.empty:
                    local_X_img = stage.transform(local_X_img)

            local_X = {TABULAR_KEY: local_X_tab, IMAGE_KEY: local_X_img}

            # Fit the model
            results = self.stages[-1].predict(local_X, *args, **kwargs)

            return self.output(results)

    else:
        raise ValueError(f"Multimodal Type Unknown {multimodal_type}")

    return predict_impl


def _generate_predict_proba(multimodal_type) -> Callable:

    if multimodal_type is None:

        @decorators.benchmark
        def predict_proba_impl(
            self: Any, X: pd.DataFrame, *args: Any, **kwargs: Any
        ) -> pd.DataFrame:

            local_X = X.copy()

            for stage in self.stages[:-1]:
                local_X = pd.DataFrame(local_X)
                local_X = stage.transform(local_X)

            result = self.stages[-1].predict_proba(local_X, *args, **kwargs)

            if result.isnull().values.any():
                raise ValueError(
                    "pipeline ({})({}) failed: nan in predict_proba output".format(
                        self.name(), self.get_args()
                    )
                )
            return self.output(result)

    elif multimodal_type == "late_fusion":

        @decorators.benchmark
        def predict_proba_impl(
            self: Any, X: pd.DataFrame, *args: Any, **kwargs: Any
        ) -> pd.DataFrame:

            if isinstance(X, dict):
                modality = self.stages[-1].modality_type()
                local_X = X[modality].copy()
            else:
                local_X = X.copy()

            for stage in self.stages[:-1]:
                local_X = pd.DataFrame(local_X)
                local_X = stage.transform(local_X)

            result = self.stages[-1].predict_proba(local_X, *args, **kwargs)

            if result.isnull().values.any():
                raise ValueError(
                    "pipeline ({})({}) failed: nan in predict_proba output".format(
                        self.name(), self.get_args()
                    )
                )
            return self.output(result)

    elif multimodal_type == "early_fusion":

        @decorators.benchmark
        def predict_proba_impl(
            self: Any, X: pd.DataFrame, *args: Any, **kwargs: Any
        ) -> pd.DataFrame:

            local_X_tab = X[TABULAR_KEY].copy()
            local_X_img = X[IMAGE_KEY].copy()

            if X.get("lr", None) is None:
                # Process images
                for stage in self.stages[:-2]:
                    local_X_img = pd.DataFrame(local_X_img)
                    if stage.modality_type() == IMAGE_KEY and not local_X_img.empty:
                        local_X_img = stage.transform(local_X_img, *args)
            else:
                local_X_img = X["lr"]

            # Process tabular
            for stage in self.stages[:-2]:
                local_X_tab = pd.DataFrame(local_X_tab)
                if stage.modality_type() == TABULAR_KEY and not local_X_tab.empty:
                    local_X_tab = stage.transform(local_X_tab)

            local_X = {TABULAR_KEY: local_X_tab, IMAGE_KEY: local_X_img}

            # Fusion plugin
            local_X = self.stages[-2].transform(local_X)

            # Predict Proba
            result = self.stages[-1].predict_proba(local_X, *args, **kwargs)

            if result.isnull().values.any():
                raise ValueError(
                    "pipeline ({})({}) failed: nan in predict_proba output".format(
                        self.name(), self.get_args()
                    )
                )
            return self.output(result)

    elif multimodal_type == "intermediate_fusion":

        @decorators.benchmark
        def predict_proba_impl(
            self: Any, X: pd.DataFrame, *args: Any, **kwargs: Any
        ) -> pd.DataFrame:

            local_X_tab = X[TABULAR_KEY].copy()
            local_X_img = X[IMAGE_KEY].copy()

            for stage in self.stages[:-1]:
                local_X_tab = pd.DataFrame(local_X_tab)
                local_X_img = pd.DataFrame(local_X_img)
                if stage.modality_type() == TABULAR_KEY and not local_X_tab.empty:
                    local_X_tab = stage.transform(local_X_tab)
                elif stage.modality_type() == IMAGE_KEY and not local_X_img.empty:
                    local_X_img = stage.transform(local_X_img)

            local_X = {TABULAR_KEY: local_X_tab, IMAGE_KEY: local_X_img}

            result = self.stages[-1].predict_proba(local_X, *args, **kwargs)

            if result.isnull().values.any():
                raise ValueError(
                    "pipeline ({})({}) failed: nan in predict_proba output".format(
                        self.name(), self.get_args()
                    )
                )
            return self.output(result)

    else:
        raise ValueError(f"Multimodal Type Unknown {multimodal_type}")

    return predict_proba_impl


def _generate_score() -> Callable:
    @decorators.benchmark
    def predict_score(self: Any, X: pd.DataFrame, y: pd.DataFrame) -> float:

        local_X = X.copy()

        for stage in self.stages[:-1]:
            local_X = stage.transform(local_X)

        if isinstance(X, list):
            local_X = [local_X, X[1]]

        return self.stages[-1].score(local_X, y)

    return predict_score


def _generate_save_template() -> Callable:
    def save_template_impl(self: Any) -> bytes:
        plugins = []
        for plugin in self.plugin_types:
            plugins.append(plugin.fqdn())

        buff = {
            "args": self.args,
            "plugins": plugins,
            "name": self.name(),
        }
        return serialization.save(buff)

    return save_template_impl


def _generate_save() -> Callable:
    def save_impl(self: Any) -> bytes:
        plugins = []
        for stage in self.stages:
            plugins.append(stage.fqdn())

        buff = self.__dict__.copy()

        buff["plugins"] = plugins

        return serialization.save(buff)

    return save_impl


def _generate_load_template() -> Callable:
    def load_template_impl(buff: bytes) -> dict:
        return serialization.load(buff)

    return load_template_impl


def _generate_load() -> Callable:
    def load_impl(buff: bytes) -> dict:
        return serialization.load(buff)

    return load_impl


def _generate_setstate() -> Callable:
    def setstate_impl(self: Any, state: dict) -> Any:
        repr_ = serialization.load(state["object"])

        self.__dict__ = repr_

        return self

    return setstate_impl


def _generate_getstate() -> Callable:
    def getstate_impl(self: Any) -> dict:
        return {"object": self.save()}

    return getstate_impl


def _modality_type() -> Callable:
    def modality_type_implt(self: Any) -> bool:
        return self.stages[-1].modality_type()

    return modality_type_implt


def _get_classifier() -> Callable:
    def get_image_model(self: Any):
        return self.stages[-1]

    return get_image_model


def _get_multimodal_type() -> Callable:
    def get_multimodal_type(self: Any):
        return self.multimodal_type

    return get_multimodal_type


__all__ = [
    "_generate_name_impl",
    "_generate_type_impl",
    "_generate_hyperparameter_space_impl",
    "_generate_hyperparameter_space_for_layer_impl",
    "_generate_sample_param_impl",
    "_generate_constructor",
    "_generate_fit",
    "_generate_is_fitted",
    "_generate_predict",
    "_generate_predict_proba",
    "_generate_score",
    "_generate_get_args",
    "_generate_load_template",
    "_generate_load",
    "_generate_save_template",
    "_generate_save",
    "_generate_setstate",
    "_generate_getstate",
    "_generate_change_output",
    "_modality_type",
    "_get_classifier",
    "_get_multimodal_type",
]
