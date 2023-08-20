# This figure will provide:
# - a table that shows the results improvements when using intermediate fusion (with all features)
# - a plot which shows how each individual features improve the model accuracy
# - investigate if and how the addition of another features influences the explainability of our models

# stdlib
import os

# third party
# Third-Party
import psutil

# autoprognosis absolute
from autoprognosis.studies.multimodal_classifier import MultimodalStudy

from tmp_lucas.loader import DataLoader

if __name__ == "__main__":
    # Train early fusion model and select only one feature at a time
    # Don't forget to predefined the optimal representation in multimodal classifier combos.

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

    df = DL.load_dataset(raw=False, sample=False, pacheco=True, full_size=False)

    print("Loaded Images")

    predefined_cnn = ["alexnet"]
    multimodal_type = "early_fusion"
    dim_red = ["cnn_fine_tune"]
    classifier = ["neural_nets"]
    unique_variable = [
        "background_father",
        "age",
        "diameter_1",
        "diameter_2",
        "smoke",
        "drink",
        "background_mother",
        "pesticide",
        "gender",
        "skin_cancer_history",
        "cancer_history",
        "has_piped_water",
        "has_sewage_system",
        "region",
        "itch",
        "grew",
        "hurt",
        "changed",
        "bleed",
        "elevation",
        "fitspatrick",
    ]

    results = {}

    for column in unique_variable:
        if column in ["image", "label"]:
            continue

        results[column] = {"auc": None, "acc": None, "bacc": None}

        study_name = (
            f"early_fusion_{dim_red[0]}_{predefined_cnn[0]}_{classifier[0]}_{column}"
        )

        variable = []
        for column_one_hot in df.columns:
            if column + "_" in column_one_hot or column == column_one_hot:
                variable.append(column_one_hot)

        # Select the variable you would like to test
        df = df[["image", "label"] + variable]

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
                num_iter=100,
                metric="aucroc",
                classifiers=classifier,
                timeout=int(3000 * 3600),
                num_study_iter=1,
                workspace="tmp_early_fusion/",
            )

            model, metrics = study.run()

            for metric, value in metrics["str"].items():
                if metric == "aucroc":
                    results[column]["aucroc"] = value
                elif metric == "accuracy":
                    results[column]["accuracy"] = value
                elif metric == "balanced_accuracy":
                    results[column]["accuracy"] = value

            print(results)

        # Here we get a model and we can evaluate estimator of the model
        # But we could also just 21 - optimization
        #  - this means 21 * 5 cnn fitting
        #  - this 100 * 21 * 5 neural nets

    # The figure would be a plot with unique variable on the bottom and their cross-validation accuracy
