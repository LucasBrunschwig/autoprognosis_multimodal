# stdlib
import json

# autoprognosis absolute
from autoprognosis.explorers.core.selector import PipelineSelector
from autoprognosis.plugins.ensemble.classifiers import WeightedEnsemble


def build_model_pipeline(
    prediction,
    calibration=[],
    imputers=[],
    feature_scaling=[],
    feature_selection=[],
    preprocess_images=False,
    image_processing=[],
    image_dimensionality_reduction=[],
    fusion=[],
    classifier_category="classifier",
):

    pipeline = PipelineSelector(
        prediction,
        calibration,
        imputers,
        feature_scaling,
        feature_selection,
        preprocess_images,
        image_processing,
        image_dimensionality_reduction,
        fusion,
        classifier_category,
    )

    return pipeline


def build_intermediate_fusion_from_dict(params_):
    # Select the pipeline parameters
    pipeline_params_name = [
        "prediction",
        "imputers",
        "feature_selection",
        "feature_scaling",
        "image_preprocessing",
        "image_dimensionality_reduction",
        "classifier_category",
    ]

    pipeline_params = {}
    for name, value in params_.items():
        if name in pipeline_params_name:
            if name == "prediction":
                pipeline_params[name] = value["name"]
            else:
                pipeline_params[name] = [value["name"]]

    # Build the image pipeline model
    pipeline = build_model_pipeline(**pipeline_params)
    potential_params = pipeline.hyperparameter_space()

    # Select the correct params
    named_params = {}
    for name in pipeline_params.keys():
        params = params_[name]["params"]
        for param in potential_params:
            named_param = param.name
            if (
                named_param.split(".")[-1] in params.keys()
                and named_param.split(".")[0] == name
            ):
                named_params[named_param] = params[named_param.split(".")[-1]]

    estimator_ = pipeline.get_multimodal_pipeline_from_named_args(**named_params)

    return estimator_


def build_image_from_dict(params_):
    # Select the pipeline parameters
    pipeline_params_name = [
        "prediction",
        "imputers",
        "feature_selection",
        "feature_scaling",
        "image_preprocessing",
        "image_dimensionality_reduction",
        "classifier_category",
    ]

    pipeline_params = {}
    for name, value in params_.items():
        if name in pipeline_params_name:
            if name == "prediction":
                pipeline_params[name] = value["name"]
            else:
                pipeline_params[name] = [value["name"]]

    # Build the image pipeline model
    pipeline = build_model_pipeline(**pipeline_params)
    potential_params = pipeline.hyperparameter_space()

    # Select the correct params
    named_params = {}
    for name in pipeline_params.keys():
        params = params_[name]["params"]
        for param in potential_params:
            named_param = param.name
            if (
                named_param.split(".")[-1] in params.keys()
                and named_param.split(".")[0] == name
            ):
                named_params[named_param] = params[named_param.split(".")[-1]]

    estimator_ = pipeline.get_image_pipeline_from_named_args(**named_params)

    return estimator_


def build_late_fusion_from_dict(params_):

    # Select the pipeline parameters
    pipeline_params_name = [
        "prediction",
        "imputers",
        "feature_selection",
        "feature_scaling",
        "image_preprocessing",
        "image_dimensionality_reduction",
        "classifier_category",
    ]

    image_params = params_["image"]
    pipeline_params = {}
    for name, value in image_params.items():
        if name in pipeline_params_name:
            if name == "prediction":
                pipeline_params[name] = value["name"]
            else:
                pipeline_params[name] = [value["name"]]

    # Build the image pipeline model
    pipeline = build_model_pipeline(**pipeline_params)
    potential_params = pipeline.hyperparameter_space()

    # Select the correct params
    named_params = {}
    for name in pipeline_params.keys():
        params = params_["image"][name]["params"]
        for param in potential_params:
            named_param = param.name
            if (
                named_param.split(".")[-1] in params.keys()
                and named_param.split(".")[0] == name
            ):
                named_params[named_param] = params[named_param.split(".")[-1]]

    image_model = pipeline.get_image_pipeline_from_named_args(**named_params)

    # Select the correct params
    tabular_params = params_["tabular"]
    pipeline_params = {}
    for name, value in tabular_params.items():
        if name in pipeline_params_name:
            if name == "prediction":
                pipeline_params[name] = value["name"]
            else:
                pipeline_params[name] = [value["name"]]

    # Build the image pipeline model
    pipeline = build_model_pipeline(**pipeline_params)
    potential_params = pipeline.hyperparameter_space()

    named_params = {}
    for name in pipeline_params.keys():
        params = params_["tabular"][name]["params"]
        for param in potential_params:
            named_param = param.name
            if (
                named_param.split(".")[-1] in params.keys()
                and named_param.split(".")[0] == name
            ):
                named_params[named_param] = params[named_param.split(".")[-1]]

    tabular_model = pipeline.get_pipeline_from_named_args(**named_params)

    if params_["ensemble"]["type"] == "weighted":
        estimator = WeightedEnsemble(
            models=[tabular_model, image_model],
            weights=params_["ensemble"]["params"]["weights"],
        )
    else:
        estimator = None

    return estimator


if __name__ == "__main__":
    with open("../config/test_late_fusion.json", "r") as file:
        params_dict = json.load(file)

    estimator_late = build_late_fusion_from_dict(params_dict)

    with open("../config/test_intermediate_fusion.json", "r") as file:
        params_dict = json.load(file)

    estimator_intermediate = build_intermediate_fusion_from_dict(params_dict)
