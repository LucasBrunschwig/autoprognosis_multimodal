# This figure will provide:
# - a table that shows the results improvements when using intermediate fusion (with all features)
# - a plot which shows how each individual features improve the model accuracy
# - investigate if and how the addition of another features influences the explainability of our models

# stdlib
import json
import os

# third party
import pandas as pd
import psutil

# autoprognosis absolute
from autoprognosis.explorers.core.defaults import (
    default_feature_scaling_names,
    default_feature_selection_names,
    default_fusion,
)
from autoprognosis.explorers.core.selector import PipelineSelector
from autoprognosis.studies.multimodal_classifier import MultimodalStudy
from autoprognosis.utils.default_modalities import dataset_to_multimodal
from autoprognosis.utils.tester import evaluate_multimodal_estimator

from tmp_lucas.loader import DataLoader


def build_pipeline(classifier, multimodal_type):
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


if __name__ == "__main__":

    train_model = False
    predefined_model = "../config/alexnet_early_fusion_nn_high.json"

    results_dir = "figure_output/"
    os.makedirs(results_dir, exist_ok=True)

    print("Loading Images")

    print(
        f"GB available before loading data: {psutil.virtual_memory().available / 1073741824:.2f}"
    )

    DL = DataLoader(
        path_="../../data",
        data_src_="PAD-UFES",
        format_="PIL",
    )

    df_train, df_test = DL.load_dataset(
        raw=False, sample=False, pacheco=False, full_size=True
    )
    group = ["_".join(patient.split("_")[0:2]) for patient in list(df_train.index)]
    df_train["patient"] = group
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    print("Loaded Images")

    predefined_cnn = ["alexnet"]
    multimodal_type = "early_fusion"

    if multimodal_type == "intermediate_fusion":
        dim_red = []
        classifier = ["intermediate_conv_net"]
        study_name = f"intermediate_fusion_{classifier}_{predefined_cnn[0]}"

    elif multimodal_type == "early_fusion":
        dim_red = ["cnn_fine_tune"]
        classifier = "neural_nets"
        study_name = f"early_fusion_{dim_red[0]}_{predefined_cnn[0]}_{classifier[0]}_{classifier[1]}"

    if train_model:
        print("Started Training")
        study = MultimodalStudy(
            study_name=study_name,
            dataset=df_train,  # pandas DataFrame
            multimodal_type=multimodal_type,
            image="image",
            target="label",  # the label column in the dataset
            sample_for_search=False,  # no Sampling
            predefined_cnn=predefined_cnn,
            feature_selection=[],
            image_processing=[],
            image_dimensionality_reduction=dim_red,
            imputers=["ice"],
            n_folds_cv=5,
            num_iter=200,
            metric="aucroc",
            classifiers=classifier,
            timeout=int(3000 * 3600),
            num_study_iter=1,
            workspace="tmp_intermediate/",
            group_id="patient",
            random_state=8,
        )

        study.run()

    elif predefined_model is not None:
        with open(predefined_model, "r") as file:
            predefined_model = json.load(file)

        # does not work because we need normalizer and imputation parameters
        model = build_pipeline(classifier, "early_fusion")
        model = model.get_multimodal_pipeline_from_named_args(**predefined_model)
        df_train = df_train.drop(["patient"], axis=1)

        X, y = dataset_to_multimodal(df_train, image=["image"], label="label")
        results = evaluate_multimodal_estimator(
            X=X,
            Y=y,
            estimator=model,
            multimodal_type="intermediate_fusion",
            n_folds=5,
            group_ids=pd.Series(group),
            seed=8,
        )

        for metric, value in results["str"].items():
            print(metric, value)
