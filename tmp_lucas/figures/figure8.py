# TRAIN AND TEST METRICS OF BEST CONFIGURATION OF MODELS
# stdlib
import json
import os

# third party
import pandas as pd
import psutil
from sklearn.preprocessing import LabelEncoder

# autoprognosis absolute
from autoprognosis.explorers.core.defaults import (
    default_feature_scaling_names,
    default_feature_selection_names,
    default_fusion,
)
from autoprognosis.explorers.core.selector import PipelineSelector
from autoprognosis.plugins.ensemble.classifiers import WeightedEnsemble
from autoprognosis.utils.tester import evaluate_multimodal_estimator

from tmp_lucas import DataLoader


def build_pipeline(classifier, multimodal_type):

    if multimodal_type == "intermediate_fusion":
        return PipelineSelector(
            classifier=classifier,
            image_processing=[],
            imputers=["ice"],
            image_dimensionality_reduction=[],
            feature_scaling=default_feature_scaling_names,
            feature_selection=default_feature_selection_names,
            multimodal_type=multimodal_type,
            fusion=[],
        )
    else:
        return PipelineSelector(
            classifier=classifier,
            image_processing=[],
            imputers=["ice"],
            image_dimensionality_reduction=["cnn_fine_tune"],
            feature_scaling=default_feature_scaling_names,
            feature_selection=default_feature_selection_names,
            multimodal_type=multimodal_type,
            fusion=default_fusion,
        )


def build_multimodal_dataset(df):
    df.reset_index(drop=True, inplace=True)
    df_label = df.label
    df = df.drop(["label"], axis=1)
    df_features = {"tab": df[df.columns.difference(["image"])], "img": df[["image"]]}
    return df_features, df_label


if __name__ == "__main__":

    model_output = "tmp_final/"
    os.makedirs(model_output, exist_ok=True)

    fit = False

    print("Loading Data")

    print(
        f"GB available before loading data: {psutil.virtual_memory().available / 1073741824:.2f}"
    )

    DL = DataLoader(
        path_="../../data",
        data_src_="PAD-UFES",
        format_="PIL",
    )

    df_train, df_test = DL.load_dataset(
        raw=False, sample=False, pacheco=True, full_size=False, size=400
    )
    group = ["_".join(patient.split("_")[0:2]) for patient in list(df_train.index)]

    df_train_features, df_train_label = build_multimodal_dataset(df_train)
    df_test_features, df_test_label = build_multimodal_dataset(df_test)

    print(
        f"GB available after loading data: {psutil.virtual_memory().available / 1073741824:.2f}"
    )

    # Encode the labels
    encoder = LabelEncoder().fit(df_train_label)
    df_train_label_encoded = pd.Series(encoder.transform(df_train_label))
    df_test_label_encoded = pd.DataFrame(encoder.transform(df_test_label))

    # Load best tabular model
    with open("../config/optimal_full/early_fusion_alexnet_nn.json", "r") as f:
        param = json.load(f)
        pipeline = build_pipeline("neural_nets", "early_fusion")
        model_logistic = pipeline.get_multimodal_pipeline_from_named_args(**param)

    with open("../config/optimal_full/early_fusion_rf.json", "r") as f:
        param = json.load(f)
        pipeline = build_pipeline("random_forest", "early_fusion")
        model_random_forest = pipeline.get_multimodal_pipeline_from_named_args(**param)

    model_early_w = WeightedEnsemble([model_logistic, model_random_forest], [0.8, 0.2])

    models = [model_logistic, model_random_forest]
    classifiers = ["early fusion neural_nets", "early fusion random_forest"]

    for model, classifier in zip(models, classifiers):
        results = evaluate_multimodal_estimator(
            model,
            df_train_features,
            df_train_label,
            n_folds=5,
            group_ids=pd.Series(group),
            seed=8,
        )

        for key, value in results["str"].items():
            print(f"f{classifier}: {key} - {value}")
