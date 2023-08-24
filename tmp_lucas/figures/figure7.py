# stdlib
import json
import os

# third party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Third-Party
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
from autoprognosis.utils.tester import classifier_metrics

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
    # In this figure we want to consider influence of specific clinical features on images.
    # For example, we trained AlexNet on Images and then we evaluate the accuracy of patient bleeding and not bleeding
    # To see if the addition of the feature bleeding helped in creating the dataset.

    results_dir = "figure_output/"
    os.makedirs(results_dir, exist_ok=True)

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
        raw=False, sample=False, pacheco=True, full_size=True
    )
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    df_test_label = df_test.label
    df_train_label = df_train.label
    df_train_features = df_train.drop(["label"], axis=1)
    df_test_features = df_test.drop(["label"], axis=1)

    print("Loaded Data")

    # Load Alexnet cnnfine-tuning optimal Architecture
    with open("../config/new_alexnet_fine_tune.json") as f:
        params = json.load(f)
    cnn_fine_tune = build_pipeline("cnn_fine_tune", None)
    cnn_fine_tune_model = cnn_fine_tune.get_image_pipeline_from_named_args(**params)

    # Load Combination of tabular classifiers
    tabular_models = []
    predefined_model = [
        "../config/neural_nets_pacheco.json",
        "../config/random_forest_pacheco.json",
        "../config/logistic_regression_pacheco.json",
    ]
    classifiers = ["neural_nets", "random_forest", "logistic_regression"]
    weights = [0.3, 0.5, 0.2]
    for file, classifier in zip(predefined_model, classifiers):
        with open(file, "r") as f:
            model_params = json.load(f)
        pipeline = build_pipeline(classifier, None)
        model = pipeline.get_pipeline_from_named_args(**model_params)
        tabular_models.append(model)
    combination_model = WeightedEnsemble(tabular_models, weights=weights)

    # Load Early Fusion Alexnet optimal with all variable in random forest
    with open("../config/alexnet_early_fusion_rf_all.json") as f:
        params = json.load(f)
    early_fusion_alexnet = build_pipeline("random_forest", "early_fusion")
    early_fusion_alexnet_model = (
        early_fusion_alexnet.get_multimodal_pipeline_from_named_args(**params)
    )

    with open("../config/ef_one_feature/bleed.json") as f:
        params = json.load(f)
    early_fusion_alexnet = build_pipeline("neural_nets", "early_fusion")
    early_fusion_bleed = early_fusion_alexnet.get_multimodal_pipeline_from_named_args(
        **params
    )

    with open("../config/ef_one_feature/age.json") as f:
        params = json.load(f)
    early_fusion_alexnet = build_pipeline("neural_nets", "early_fusion")
    early_fusion_age = early_fusion_alexnet.get_multimodal_pipeline_from_named_args(
        **params
    )

    # ---------------------------------------------------------------------------------------------------------------- #
    # Here we want to test bleeding: precision if we have only images, only images + bleeding, only images + all
    # The plot will look like a categorical where X = diagnoses, each divided in yes or no,

    evaluator = classifier_metrics("accuracy")

    # Evaluate Fine Tuning
    label_encoder = LabelEncoder().fit(df_train_label)
    df_train_label_encoded = label_encoder.transform(df_train_label)
    df_test_label_encoded = pd.DataFrame(label_encoder.transform(df_test_label))

    combination_model.fit(
        df_train_features[df_train_features.columns.difference(["image"])],
        df_train_label_encoded,
    )
    preds = combination_model.predict(
        df_train_features[df_test_features.columns.difference(["image"])]
    ).astype(int)

    accuracy_bleed = [[], [], [], [], [], []]
    size_bleed = [[], [], [], [], [], []]
    accuracy_age = [[], [], [], [], [], []]
    size_age = [[], [], [], [], [], []]
    for i, label in enumerate(df_test_label.unique()):
        class_ix = df_test[
            (df_test.bleed_True == 1) & (df_test_label == label)
        ].index.tolist()
        size_bleed[i].append(len(class_ix))
        if len(class_ix) > 0:
            metric = evaluator.score_proba(
                preds.iloc[class_ix], df_test_label_encoded.iloc[class_ix]
            )
            accuracy_bleed[i].append(metric["accuracy"])
            print("Fine Tuning Bleed = True:", metric)
        else:
            accuracy_bleed[i].append(0)
        class_ix = df_test[
            (df_test.bleed_True == 0) & (df_test_label == label)
        ].index.tolist()
        size_bleed[i].append(len(class_ix))
        if len(class_ix) > 0:
            metric = evaluator.score_proba(
                preds.iloc[class_ix], df_test_label_encoded.iloc[class_ix]
            )
            accuracy_bleed[i].append(metric["accuracy"])
            print("Fine Tuning Bleed = False:", metric)
        else:
            accuracy_bleed[i].append(0)
        class_ix = df_test[df_test_label == label].index.tolist()
        size_bleed[i].append(len(class_ix))
        if len(class_ix) > 0:
            metric = evaluator.score_proba(
                preds.iloc[class_ix], df_test_label_encoded.iloc[class_ix]
            )
            accuracy_bleed[i].append(metric["accuracy"])
            print("Fine Tuning class:", metric)
        else:
            accuracy_bleed[i].append(0)
        class_ix = df_test[
            (df_test.age >= 50) & (df_test_label == label)
        ].index.tolist()
        size_age[i].append(len(class_ix))
        if len(class_ix) > 0:
            metric = evaluator.score_proba(
                preds.iloc[class_ix], df_test_label_encoded.iloc[class_ix]
            )
            accuracy_age[i].append(metric["accuracy"])
            print("Fine Tuning Age >= 50:", metric)
        else:
            accuracy_age[i].append(0)
        class_ix = df_test[(df_test.age < 50) & (df_test_label == label)].index.tolist()
        size_age[i].append(len(class_ix))
        if len(class_ix) > 0:
            metric = evaluator.score_proba(
                preds.iloc[class_ix], df_test_label_encoded.iloc[class_ix]
            )
            accuracy_age[i].append(metric["accuracy"])
            print("Fine Tuning Age < 50:", metric)
        else:
            accuracy_age[i].append(0)
        class_ix = df_test[df_test_label == label].index.tolist()
        size_age[i].append(len(class_ix))
        if len(class_ix) > 0:
            metric = evaluator.score_proba(
                preds.iloc[class_ix], df_test_label_encoded.iloc[class_ix]
            )
            accuracy_age[i].append(metric["accuracy"])
            print("Fine Tuning class:", metric)
        else:
            accuracy_age[i].append(0)

    x = np.arange(len(label_encoder.classes_))
    width = 0.2  # the width of the bars
    colors = ["blue", "red", "green"]

    labels = ["bleed", "no bleed", "all"]
    labels_classes = []
    for i, cla in enumerate(label_encoder.classes_):
        labels_classes.append(
            f"{cla} ({size_bleed[i][0]},{size_bleed[i][1]},{size_bleed[i][2]})"
        )

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, value in enumerate(accuracy_bleed):
        for j, (v, color) in enumerate(zip(value, colors)):
            ax.bar(
                x[i] + j * width - width,
                v,
                width,
                label=labels[j] if i == 0 else "",
                color=color,
            )
    plt.legend()
    ax.set_xticks(x, labels_classes)
    ax.set_ylim(0, 1)
    plt.savefig("figure_output/tabular_bleed_analysis.png")

    x = np.arange(len(label_encoder.classes_))
    width = 0.2  # the width of the bars
    colors = ["blue", "red", "green"]
    labels_classes = []
    for i, cla in enumerate(label_encoder.classes_):
        labels_classes.append(
            f"{cla} ({size_age[i][0]},{size_age[i][1]},{size_age[i][2]}"
        )
    labels = ["age >= 50", "age < 50", "all"]
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, value in enumerate(accuracy_age):
        for j, (v, color) in enumerate(zip(value, colors)):
            ax.bar(
                x[i] + j * width - width,
                v,
                width,
                label=labels[j] if i == 0 else "",
                color=color,
            )
    plt.legend()
    ax.set_xticks(x, labels_classes)
    ax.set_ylim(0, 1)
    plt.savefig("figure_output/tabular_age_analysis.png")

    cnn_fine_tune_model.fit(df_train_features[["image"]], df_train_label_encoded)
    preds = cnn_fine_tune_model.predict(df_test_features[["image"]]).astype(int)

    accuracy_bleed = [[], [], [], [], [], []]
    size_bleed = [[], [], [], [], [], []]
    accuracy_age = [[], [], [], [], [], []]
    size_age = [[], [], [], [], [], []]
    for i, label in enumerate(df_test_label.unique()):
        class_ix = df_test[
            (df_test.bleed_True == 1) & (df_test_label == label)
        ].index.tolist()
        size_bleed[i].append(len(class_ix))
        if len(class_ix) > 0:
            metric = evaluator.score_proba(
                preds.iloc[class_ix], df_test_label_encoded.iloc[class_ix]
            )
            accuracy_bleed[i].append(metric["accuracy"])
            print("Fine Tuning Bleed = True:", metric)
        else:
            accuracy_bleed[i].append(0)
        class_ix = df_test[
            (df_test.bleed_True == 0) & (df_test_label == label)
        ].index.tolist()
        size_bleed[i].append(len(class_ix))
        if len(class_ix) > 0:
            metric = evaluator.score_proba(
                preds.iloc[class_ix], df_test_label_encoded.iloc[class_ix]
            )
            accuracy_bleed[i].append(metric["accuracy"])
            print("Fine Tuning Bleed = False:", metric)
        else:
            accuracy_bleed[i].append(0)
        class_ix = df_test[df_test_label == label].index.tolist()
        size_bleed[i].append(len(class_ix))

        if len(class_ix) > 0:
            metric = evaluator.score_proba(
                preds.iloc[class_ix], df_test_label_encoded.iloc[class_ix]
            )
            accuracy_bleed[i].append(metric["accuracy"])
            print("Fine Tuning class:", metric)
        else:
            accuracy_bleed[i].append(0)
        class_ix = df_test[
            (df_test.age >= 50) & (df_test_label == label)
        ].index.tolist()
        size_age[i].append(len(class_ix))
        if len(class_ix) > 0:
            metric = evaluator.score_proba(
                preds.iloc[class_ix], df_test_label_encoded.iloc[class_ix]
            )
            accuracy_age[i].append(metric["accuracy"])
            print("Fine Tuning Age > 50:", metric)
        else:
            accuracy_age[i].append(0)
        class_ix = df_test[(df_test.age < 50) & (df_test_label == label)].index.tolist()
        size_age[i].append(len(class_ix))
        if len(class_ix) > 0:
            metric = evaluator.score_proba(
                preds.iloc[class_ix], df_test_label_encoded.iloc[class_ix]
            )
            accuracy_age[i].append(metric["accuracy"])
            print("Fine Tuning Age < 50:", metric)
        else:
            accuracy_age[i].append(0)
        class_ix = df_test[df_test_label == label].index.tolist()
        size_age[i].append(len(class_ix))
        if len(class_ix) > 0:
            metric = evaluator.score_proba(
                preds.iloc[class_ix], df_test_label_encoded.iloc[class_ix]
            )
            accuracy_age[i].append(metric["accuracy"])
            print("Fine Tuning class:", metric)
        else:
            accuracy_age[i].append(0)

    x = np.arange(len(label_encoder.classes_))
    width = 0.2  # the width of the bars
    colors = ["blue", "red", "green"]

    labels = ["bleed", "no bleed", "all"]
    labels_classes = []
    for i, cla in enumerate(label_encoder.classes_):
        labels_classes.append(
            f"{cla} ({size_bleed[i][0]},{size_bleed[i][1]},{size_bleed[i][2]}"
        )

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, value in enumerate(accuracy_bleed):
        for j, (v, color) in enumerate(zip(value, colors)):
            ax.bar(
                x[i] + j * width - width,
                v,
                width,
                label=labels[j] if i == 0 else "",
                color=color,
            )
    plt.legend()
    ax.set_xticks(x, labels_classes)
    ax.set_ylim(0, 1)
    plt.savefig("figure_output/cnn_fine_tune_bleed_analysis.png")

    x = np.arange(len(label_encoder.classes_))
    width = 0.2  # the width of the bars
    colors = ["blue", "red", "green"]
    labels_classes = []
    for i, cla in enumerate(label_encoder.classes_):
        labels_classes.append(
            f"{cla} ({size_age[i][0]},{size_age[i][1]},{size_age[i][2]}"
        )
    labels = ["age >= 50", "age < 50", "all"]
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, value in enumerate(accuracy_age):
        for j, (v, color) in enumerate(zip(value, colors)):
            ax.bar(
                x[i] + j * width - width,
                v,
                width,
                label=labels[j] if i == 0 else "",
                color=color,
            )
    plt.legend()
    ax.set_xticks(x, labels_classes)
    ax.set_ylim(0, 1)
    plt.savefig("figure_output/cnn_fine_tune_age_analysis.png")

    # Evaluate Early Fusion
    dict_df_train = {
        "tab": df_train_features[df_train_features.columns.difference(["image"])],
        "img": df_train_features[["image"]],
    }
    dict_df_test = {
        "tab": df_test_features[df_test_features.columns.difference(["image"])],
        "img": df_test_features[["image"]],
    }
    early_fusion_alexnet_model.fit(dict_df_train, pd.DataFrame(df_train_label_encoded))
    preds = early_fusion_alexnet_model.predict(dict_df_test).astype(int)

    accuracy_bleed = [[], [], [], [], [], []]
    size_bleed = [[], [], [], [], [], []]
    accuracy_age = [[], [], [], [], [], []]
    size_age = [[], [], [], [], [], []]
    for i, label in enumerate(df_test_label.unique()):
        class_ix = df_test[
            (df_test.bleed_True == 1) & (df_test_label == label)
        ].index.tolist()
        size_bleed[i].append(len(class_ix))
        if len(class_ix) > 0:
            metric = evaluator.score_proba(
                preds.iloc[class_ix], df_test_label_encoded.iloc[class_ix]
            )
            accuracy_bleed[i].append(metric["accuracy"])
            print("Fine Tuning Bleed = True:", metric)
        else:
            accuracy_bleed[i].append(0)
        class_ix = df_test[
            (df_test.bleed_True == 0) & (df_test_label == label)
        ].index.tolist()
        size_bleed[i].append(len(class_ix))
        if len(class_ix) > 0:
            metric = evaluator.score_proba(
                preds.iloc[class_ix], df_test_label_encoded.iloc[class_ix]
            )
            accuracy_bleed[i].append(metric["accuracy"])
            print("Fine Tuning Bleed = False:", metric)
        else:
            accuracy_bleed[i].append(0)
        class_ix = df_test[df_test_label == label].index.tolist()
        size_bleed[i].append(len(class_ix))

        if len(class_ix) > 0:
            metric = evaluator.score_proba(
                preds.iloc[class_ix], df_test_label_encoded.iloc[class_ix]
            )
            accuracy_bleed[i].append(metric["accuracy"])
            print("Fine Tuning class:", metric)
        else:
            accuracy_bleed[i].append(0)
        class_ix = df_test[
            (df_test.age >= 50) & (df_test_label == label)
        ].index.tolist()
        size_age[i].append(len(class_ix))
        if len(class_ix) > 0:
            metric = evaluator.score_proba(
                preds.iloc[class_ix], df_test_label_encoded.iloc[class_ix]
            )
            accuracy_age[i].append(metric["accuracy"])
            print("Fine Tuning Age > 50:", metric)
        else:
            accuracy_age[i].append(0)
        class_ix = df_test[(df_test.age < 50) & (df_test_label == label)].index.tolist()
        size_age[i].append(len(class_ix))
        if len(class_ix) > 0:
            metric = evaluator.score_proba(
                preds.iloc[class_ix], df_test_label_encoded.iloc[class_ix]
            )
            accuracy_age[i].append(metric["accuracy"])
            print("Fine Tuning Age < 50:", metric)
        else:
            accuracy_age[i].append(0)
        class_ix = df_test[df_test_label == label].index.tolist()
        size_age[i].append(len(class_ix))
        if len(class_ix) > 0:
            metric = evaluator.score_proba(
                preds.iloc[class_ix], df_test_label_encoded.iloc[class_ix]
            )
            accuracy_age[i].append(metric["accuracy"])
            print("Fine Tuning class:", metric)
        else:
            accuracy_age[i].append(0)

    x = np.arange(len(label_encoder.classes_))
    width = 0.2  # the width of the bars
    colors = ["blue", "red", "green"]

    labels = ["bleed", "no bleed", "all"]
    labels_classes = []
    for i, cla in enumerate(label_encoder.classes_):
        labels_classes.append(
            f"{cla} ({size_bleed[i][0]},{size_bleed[i][1]},{size_bleed[i][2]}"
        )

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, value in enumerate(accuracy_bleed):
        for j, (v, color) in enumerate(zip(value, colors)):
            ax.bar(
                x[i] + j * width - width,
                v,
                width,
                label=labels[j] if i == 0 else "",
                color=color,
            )
    plt.legend()
    ax.set_xticks(x, labels_classes)
    ax.set_ylim(0, 1)
    plt.savefig("figure_output/early_fusion_bleed_analysis.png")

    x = np.arange(len(label_encoder.classes_))
    width = 0.2  # the width of the bars
    colors = ["blue", "red", "green"]
    labels_classes = []
    for i, cla in enumerate(label_encoder.classes_):
        labels_classes.append(
            f"{cla} ({size_age[i][0]},{size_age[i][1]},{size_age[i][2]}"
        )
    labels = ["age >= 50", "age < 50", "all"]
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, value in enumerate(accuracy_age):
        for j, (v, color) in enumerate(zip(value, colors)):
            ax.bar(
                x[i] + j * width - width,
                v,
                width,
                label=labels[j] if i == 0 else "",
                color=color,
            )
    plt.legend()
    ax.set_xticks(x, labels_classes)
    ax.set_ylim(0, 1)
    plt.savefig("figure_output/early_fusion_age_analysis.png")

    # ---------------------------------------------------------------------------------------------------------------- #

    # Evaluate Early Fusion
    dict_df_train = {
        "tab": df_train_features[["bleed_True", "bleed_False", "bleed_UNK"]],
        "img": df_train_features[["image"]],
    }
    dict_df_test = {
        "tab": df_test_features[["bleed_True", "bleed_False", "bleed_UNK"]],
        "img": df_test_features[["image"]],
    }
    early_fusion_bleed.fit(dict_df_train, pd.DataFrame(df_train_label_encoded))
    preds = early_fusion_bleed.predict(dict_df_test).astype(int)

    accuracy_bleed = [[], [], [], [], [], []]
    size_bleed = [[], [], [], [], [], []]
    for i, label in enumerate(df_test_label.unique()):
        class_ix = df_test[
            (df_test.bleed_True == 1) & (df_test_label == label)
        ].index.tolist()
        size_bleed[i].append(len(class_ix))
        if len(class_ix) > 0:
            metric = evaluator.score_proba(
                preds.iloc[class_ix], df_test_label_encoded.iloc[class_ix]
            )
            accuracy_bleed[i].append(metric["accuracy"])
            print("Fine Tuning Bleed = True:", metric)
        else:
            accuracy_bleed[i].append(0)
        class_ix = df_test[
            (df_test.bleed_True == 0) & (df_test_label == label)
        ].index.tolist()
        size_bleed[i].append(len(class_ix))
        if len(class_ix) > 0:
            metric = evaluator.score_proba(
                preds.iloc[class_ix], df_test_label_encoded.iloc[class_ix]
            )
            accuracy_bleed[i].append(metric["accuracy"])
            print("Fine Tuning Bleed = False:", metric)
        else:
            accuracy_bleed[i].append(0)
        class_ix = df_test[df_test_label == label].index.tolist()
        size_bleed[i].append(len(class_ix))

        if len(class_ix) > 0:
            metric = evaluator.score_proba(
                preds.iloc[class_ix], df_test_label_encoded.iloc[class_ix]
            )
            accuracy_bleed[i].append(metric["accuracy"])
            print("Fine Tuning class:", metric)
        else:
            accuracy_bleed[i].append(0)

    x = np.arange(len(label_encoder.classes_))
    width = 0.2  # the width of the bars
    colors = ["blue", "red", "green"]

    labels = ["bleed", "no bleed", "all"]
    labels_classes = []
    for i, cla in enumerate(label_encoder.classes_):
        labels_classes.append(
            f"{cla} ({size_bleed[i][0]},{size_bleed[i][1]},{size_bleed[i][2]}"
        )

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, value in enumerate(accuracy_bleed):
        for j, (v, color) in enumerate(zip(value, colors)):
            ax.bar(
                x[i] + j * width - width,
                v,
                width,
                label=labels[j] if i == 0 else "",
                color=color,
            )
    plt.legend()
    ax.set_xticks(x, labels_classes)
    ax.set_ylim(0, 1)
    plt.savefig("figure_output/early_fusion_one_feature_bleed_analysis.png")

    # Evaluate Early Fusion
    dict_df_train = {
        "tab": df_train_features[["age"]],
        "img": df_train_features[["image"]],
    }
    dict_df_test = {
        "tab": df_test_features[["age"]],
        "img": df_test_features[["image"]],
    }
    early_fusion_age.fit(dict_df_train, pd.DataFrame(df_train_label_encoded))
    preds = early_fusion_age.predict(dict_df_test).astype(int)

    accuracy_age = [[], [], [], [], [], []]
    size_age = [[], [], [], [], [], []]
    for i, label in enumerate(df_test_label.unique()):
        class_ix = df_test[
            (df_test.age >= 50) & (df_test_label == label)
        ].index.tolist()
        size_age[i].append(len(class_ix))
        if len(class_ix) > 0:
            metric = evaluator.score_proba(
                preds.iloc[class_ix], df_test_label_encoded.iloc[class_ix]
            )
            accuracy_age[i].append(metric["accuracy"])
            print("Fine Tuning Age > 50:", metric)
        else:
            accuracy_age[i].append(0)
        class_ix = df_test[(df_test.age < 50) & (df_test_label == label)].index.tolist()
        size_age[i].append(len(class_ix))
        if len(class_ix) > 0:
            metric = evaluator.score_proba(
                preds.iloc[class_ix], df_test_label_encoded.iloc[class_ix]
            )
            accuracy_age[i].append(metric["accuracy"])
            print("Fine Tuning Age < 50:", metric)
        else:
            accuracy_age[i].append(0)
        class_ix = df_test[df_test_label == label].index.tolist()
        size_age[i].append(len(class_ix))
        if len(class_ix) > 0:
            metric = evaluator.score_proba(
                preds.iloc[class_ix], df_test_label_encoded.iloc[class_ix]
            )
            accuracy_age[i].append(metric["accuracy"])
            print("Fine Tuning class:", metric)
        else:
            accuracy_age[i].append(0)

    x = np.arange(len(label_encoder.classes_))
    width = 0.2  # the width of the bars
    colors = ["blue", "red", "green"]
    labels_classes = []
    for i, cla in enumerate(label_encoder.classes_):
        labels_classes.append(
            f"{cla} ({size_age[i][0]},{size_age[i][1]},{size_age[i][2]}"
        )
    labels = ["age >= 50", "age < 50", "all"]
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, value in enumerate(accuracy_age):
        for j, (v, color) in enumerate(zip(value, colors)):
            ax.bar(
                x[i] + j * width - width,
                v,
                width,
                label=labels[j] if i == 0 else "",
                color=color,
            )
    plt.legend()
    ax.set_xticks(x, labels_classes)
    ax.set_ylim(0, 1)
    plt.savefig("figure_output/early_fusion_one_feature_age.png")
