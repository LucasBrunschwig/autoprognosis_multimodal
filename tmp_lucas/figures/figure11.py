# This figure check what size of the image provide the better results for the model

# In this figure we investigate the quality of the images and where the model perform poorly

# stdlib
import json
import os

# third party
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

# autoprognosis absolute
from autoprognosis.explorers.core.defaults import (
    default_feature_scaling_names,
    default_feature_selection_names,
    default_fusion,
)
from autoprognosis.explorers.core.selector import PipelineSelector

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
    results_dir = "figure_output/"
    os.makedirs(results_dir, exist_ok=True)

    DL = DataLoader(
        path_="../../data",
        data_src_="PAD-UFES",
        format_="PIL",
    )

    # Load the image model
    with open("../config/new_alexnet_fine_tune.json", "r") as f:
        param = json.load(f)
        pipeline = build_pipeline("cnn_fine_tune", "late_fusion")
        model_cnn = pipeline.get_image_pipeline_from_named_args(**param)

    # Load the intermediate model
    with open("../config/intermediate_fusion_alexnet.json", "r") as f:
        param = json.load(f)
        pipeline = build_pipeline("intermediate_conv_net", "intermediate_fusion")
        model_intermediate = pipeline.get_multimodal_pipeline_from_named_args(**param)

    with open("../config/metablock_alexnet.json", "r") as f:
        params = json.load(f)
        pipeline = build_pipeline("metablock", "intermediate_fusion")
        model_metablock = pipeline.get_multimodal_pipeline_from_named_args(**params)

    resizing = [100, 200, 300, 400, 500, 600, "original"]

    for size in resizing:

        print("# ------------- #### ------------- #")
        print(f"  Resizing: {size}")
        print("  ------")
        # Reload the images with the given size
        df_train, df_test = DL.load_dataset(
            raw=False, sample=False, pacheco=False, full_size=False, size=size
        )
        df_train_features, df_train_label = build_multimodal_dataset(df_train)
        df_test_features, df_test_label = build_multimodal_dataset(df_test)

        # Encode the labels
        encoder = LabelEncoder().fit(df_train_label)
        df_train_label_encoded = pd.Series(encoder.transform(df_train_label))
        df_test_label_encoded = pd.DataFrame(encoder.transform(df_test_label))

        # Fit the model on the training dataset
        model_cnn = model_cnn.fit(df_train_features, df_train_label_encoded)
        model_metablock = model_metablock.fit(df_train_features, df_train_label_encoded)
        model_intermediate = model_intermediate.fit(
            df_train_features, df_train_label_encoded
        )

        models = [model_cnn, model_intermediate, model_metablock]
        classifiers = ["cnn_fine_tuned", "intermediate_conv_net", "metablock"]
        for model, classifier in zip(models, classifiers):
            preds = model.predict(df_test_features)
            preds_probs = model.predict_proba(df_test_features)
            accuracy = accuracy_score(preds, df_test_label_encoded)
            print(f"  {classifier} - accuracy: {accuracy}")
            balanced = balanced_accuracy_score(preds, df_test_label_encoded)
            print(f"  {classifier} - balanced: {balanced}")
            auc_ = roc_auc_score(
                df_test_label_encoded, preds_probs, multi_class="ovr", average="micro"
            )
            print(f"  {classifier} - auc: {auc_}")
