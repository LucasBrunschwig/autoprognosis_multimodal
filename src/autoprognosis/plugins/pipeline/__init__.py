# stdlib
from typing import Any, Dict, List, Tuple, Type

# third party
from optuna.trial import Trial
import pandas as pd

# autoprognosis absolute
from autoprognosis.plugins import group

# autoprognosis relative
from .generators import (
    _generate_change_output,
    _generate_constructor,
    _generate_fit,
    _generate_get_args,
    _generate_getstate,
    _generate_hyperparameter_space_for_layer_impl,
    _generate_hyperparameter_space_impl,
    _generate_is_fitted,
    _generate_load,
    _generate_load_template,
    _generate_name_impl,
    _generate_predict,
    _generate_predict_proba,
    _generate_sample_param_impl,
    _generate_save,
    _generate_save_template,
    _generate_score,
    _generate_setstate,
    _generate_type_impl,
    _get_classifier,
    _get_multimodal_type,
    _modality_type,
)


class PipelineMeta(type):
    def __new__(cls: Type, name: str, plugins: Tuple[Type, ...], dct: dict) -> Any:

        multimodal_type = "_".join(name.split("_")[-2:])
        # if not multimodal remove the None from the name
        if "fusion" not in multimodal_type:
            multimodal_type = None
            name = "_".join(name.split("_")[:-1])

        dct["__init__"] = _generate_constructor()
        dct["__setstate__"] = _generate_setstate()
        dct["__getstate__"] = _generate_getstate()
        dct["fit"] = _generate_fit(multimodal_type)
        dct["is_fitted"] = _generate_is_fitted()
        dct["predict"] = _generate_predict(multimodal_type)
        dct["predict_proba"] = _generate_predict_proba(multimodal_type)
        dct["score"] = _generate_score()
        dct["name"] = _generate_name_impl(plugins)
        dct["type"] = _generate_type_impl(plugins)
        dct["hyperparameter_space"] = _generate_hyperparameter_space_impl(plugins)
        dct[
            "hyperparameter_space_for_layer"
        ] = _generate_hyperparameter_space_for_layer_impl(plugins)
        dct["sample_params"] = _generate_sample_param_impl(plugins)
        dct["get_args"] = _generate_get_args()

        dct["save_template"] = _generate_save_template()
        dct["save"] = _generate_save()
        dct["load"] = _generate_load()
        dct["load_template"] = _generate_load_template()
        dct["change_output"] = _generate_change_output()
        dct["modality_type"] = _modality_type()
        dct["get_classifier"] = _get_classifier()
        dct["get_multimodal_type"] = _get_multimodal_type()

        dct["plugin_types"] = list(plugins)

        return super().__new__(cls, name, tuple(), dct)

    @staticmethod
    def name(*args: Any) -> str:
        raise NotImplementedError("not implemented")

    @staticmethod
    def type(*args: Any) -> str:
        raise NotImplementedError("not implemented")

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> Dict:
        raise NotImplementedError("not implemented")

    @staticmethod
    def hyperparameter_space_for_layer(name: str, *args: Any, **kwargs: Any) -> Dict:
        raise NotImplementedError("not implemented")

    def sample_params(trial: Trial, *args: Any, **kwargs: Any) -> Dict:
        raise NotImplementedError("not implemented")

    def get_args(*args: Any, **kwargs: Any) -> Dict:
        raise NotImplementedError("not implemented")

    def fit(self: Any, X: pd.DataFrame, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("not implemented")

    def is_fitted(self: Any) -> Any:
        raise NotImplementedError("not implemented")

    def predict(*args: Any, **kwargs: Any) -> pd.DataFrame:
        raise NotImplementedError("not implemented")

    def predict_proba(*args: Any, **kwargs: Any) -> pd.DataFrame:
        raise NotImplementedError("not implemented")

    def save_template(*args: Any, **kwargs: Any) -> bytes:
        raise NotImplementedError("not implemented")

    def save(*args: Any, **kwargs: Any) -> bytes:
        raise NotImplementedError("not implemented")

    @staticmethod
    def load_template(buff: bytes) -> "PipelineMeta":
        loader = _generate_load_template()
        repr_ = loader(buff)

        template = Pipeline(repr_["plugins"])

        return template(repr_["args"])

    @staticmethod
    def load(buff: bytes) -> "PipelineMeta":
        loader = _generate_load()
        repr_ = loader(buff)

        template = Pipeline(repr_["plugins"])

        pipeline = template(repr_["args"])
        pipeline.stages = repr_["stages"]

        return pipeline

    def __getstate__(self) -> dict:
        raise NotImplementedError("not implemented")

    def __setstate__(self, state: dict) -> Any:
        raise NotImplementedError("not implemented")

    def change_output(self, output: str) -> None:
        raise NotImplementedError("not implemented")


def Pipeline(plugins_str: List[str], multimodal_type=None, data_type="tabular") -> Any:
    plugins = group(plugins_str, data_type=data_type)

    name = "_".join(p.name() for p in plugins)

    name += f"_{multimodal_type}"
    return PipelineMeta(name, plugins, {})
