# This figure will provide:
# - a table that shows the results improvements when using intermediate fusion (with all features)
# - a plot which shows how each individual features improve the model accuracy
# - investigate if and how the addition of another features influences the explainability of our models

# stdlib
import json
import os

# third party
# Third-Party
import psutil

# autoprognosis absolute
# Absolute autoprognosis
from autoprognosis.studies.multimodal_classifier import MultimodalStudy
from autoprognosis.utils.default_modalities import dataset_to_multimodal
from autoprognosis.utils.tester import evaluate_multimodal_estimator

from tmp_lucas.build_pipeline import build_intermediate_fusion_from_dict
from tmp_lucas.loader import DataLoader

if __name__ == "__main__":

    train_model = True
    predefined_model = None

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

    df = DL.load_dataset(raw=False, sample=False)

    print("Loaded Images")

    predefined_cnn = ["alexnet"]
    multimodal_type = "early_fusion"

    if multimodal_type == "intermediate_fusion":
        dim_red = []
        classifier = ["intermediate_conv_net"]
        study_name = f"test_new_intermediate_fusion_{classifier}_{predefined_cnn[0]}"

    elif multimodal_type == "early_fusion":
        dim_red = ["cnn_fine_tune"]
        classifier = ["neural_nets", "random_forest"]
        study_name = f"early_fusion_{dim_red[0]}_{predefined_cnn[0]}_{classifier[0]}_{classifier[1]}"

    if train_model:
        print("Started Training")
        study = MultimodalStudy(
            study_name=study_name,
            dataset=df,  # pandas DataFrame
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
            workspace="../",
        )

        study.run()

    elif predefined_model is not None:
        with open(predefined_model, "r") as file:
            predefined_model = json.load(file)

        # does not work because we need normalizer and imputation
        model = build_intermediate_fusion_from_dict(predefined_model)

        X, y = dataset_to_multimodal(df, image=["image"], label="label")
        results = evaluate_multimodal_estimator(
            X=X,
            Y=y,
            estimator=model,
            multimodal_type="intermediate_fusion",
            n_folds=5,
        )

    # Figure 1:
