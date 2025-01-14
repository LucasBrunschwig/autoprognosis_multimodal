# stdlib
import json
import os

# third party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import LabelEncoder

# autoprognosis absolute
from autoprognosis.explorers.core.defaults import (
    default_feature_scaling_names,
    default_feature_selection_names,
    default_fusion,
)
from autoprognosis.explorers.core.selector import PipelineSelector
from autoprognosis.plugins.ensemble.classifiers import WeightedEnsemble
from autoprognosis.studies.classifiers import ClassifierStudy
from autoprognosis.utils.serialization import load_model_from_file
from autoprognosis.utils.tester import evaluate_estimator

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


def auc_curve(preds, true):

    preds = np.array(preds)
    n_classes = true.shape[1]

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(true[:, i], preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        plt.plot(
            fpr[i], tpr[i], label=f"ROC curve of class {i} (area = {roc_auc[i]:.2f})"
        )

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) for each class")
    plt.legend(loc="lower right")
    plt.show()


if __name__ == "__main__":
    # This figure is used for Bayesian Optimization of tabular classifiers

    bayesian_optimization = False
    explain = True
    cross_validation = False

    predefined_model = []
    weights = []
    classifiers = ["logistic_regression", "random_forest", "neural_nets"]

    # Predefined model with Pacheco
    predefined_model = [
        "../config/neural_nets_pacheco.json",
        "../config/random_forest_pacheco.json",
        "../config/logistic_regression_pacheco.json",
    ]
    classifiers = ["neural_nets", "random_forest", "logistic_regression"]
    weights = [0.3, 0.5, 0.2]

    results_dir = "figure_output/"
    os.makedirs(results_dir, exist_ok=True)

    print("Loading Images")

    print(
        f"GB available before loading data: {psutil.virtual_memory().available/1073741824:.2f}"
    )

    # Use a subprocess to free memory

    DL = DataLoader(
        path_=r"../../data",
        data_src_="PAD-UFES",
        format_="PIL",
    )

    df_train, df_test = DL.load_dataset(
        raw=False, sample=False, pacheco=True, full_size=False
    )
    group = ["_".join(patient.split("_")[0:2]) for patient in list(df_train.index)]
    df_train = df_train[df_train.columns.difference(["image"])]
    df_train["patient"] = group
    df_train.reset_index(drop=True, inplace=True)
    df_test = df_test[df_test.columns.difference(["image"])]
    df_test.reset_index(drop=True, inplace=True)

    study_name = "tabular_combinations"

    if bayesian_optimization:
        study = ClassifierStudy(
            study_name=study_name,
            dataset=df_train,  # pandas DataFrame
            target="label",  # the label column in the dataset
            sample_for_search=False,  # no Sampling
            n_folds_cv=5,
            num_iter=1,
            metric="aucroc",
            classifiers=classifiers,
            timeout=int(10 * 3600),
            num_study_iter=1,
            workspace="tmp/",
            random_state=8,
            group_id="patient",
        )

        study.fit()

    if predefined_model:
        models = []
        for file, classifier in zip(predefined_model, classifiers):
            with open(file, "r") as f:
                model_params = json.load(f)

            pipeline = build_pipeline(classifier, None)
            model = pipeline.get_pipeline_from_named_args(**model_params)
            models.append(model)
        model = WeightedEnsemble(models, weights=weights)

    else:
        model = load_model_from_file("tmp/" + study_name + r"/model.p")

    if cross_validation:
        results = evaluate_estimator(
            model,
            df_train[df_train.columns.difference(["image", "label", "patient"])],
            df_train["label"],
            n_folds=5,
            group_ids=df_train["patient"],
            seed=8,
        )

        for metric, value in results["str"].items():
            print(metric, value)

    if explain:

        df_test_label = df_test.label
        df_test_features = df_test.drop(["label"], axis=1)

        df_train_label = df_train.label
        df_train_features = df_train.drop(["label", "patient"], axis=1)

        # Unique LabelEncoder
        label_encoder = LabelEncoder().fit(df_train_label)
        df_test_label = label_encoder.transform(df_test_label)
        df_train_label = label_encoder.transform(df_train_label)

        # Add an explainer and fit the model
        model.explainer_plugins = ["kernel_shap"]
        model.fit(df_train_features, df_train_label)
        predictions = model.predict(df_test_features)
        pred_probs = model.predict_proba(df_test_features)
        predictions = label_encoder.inverse_transform(predictions)
        df_test_label_num = df_test_label
        df_test_label = label_encoder.inverse_transform(df_test_label)
        # Define class names (replace with your actual class names)
        class_names = label_encoder.classes_
        class_names_str = []
        for name in class_names:
            count = pd.DataFrame(df_test_label).value_counts().loc[name].values[0]
            class_names_str.append(f"{name} ({count})")

        # Compute confusion matrix
        cm = confusion_matrix(df_test_label, predictions, normalize="true")

        # Create a heatmap plot for the confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names_str,
        )

        # Add labels, title, and axis names
        plt.xlabel("Predicted Diagnoses")
        plt.ylabel("True Diagnoses")
        plt.title("Confusion Matrix for Skin Lesion Classification")
        plt.savefig(results_dir + "confusion_matrix_combinations.png")

        accuracy = accuracy_score(predictions, df_test_label)
        print(f"accuracy: {accuracy}")
        balanced = balanced_accuracy_score(predictions, df_test_label)
        print(f"balanced: {balanced}")
        auc_ = roc_auc_score(
            df_test_label_num, pred_probs, multi_class="ovr", average="micro"
        )
        print(f"auc: {auc_}")
        f1 = f1_score(predictions, df_test_label, average="micro")
        print(f"f1: {f1}")

        # Test the explainer on test data
        model.explain_plot(df_test_features, class_names)
        plt.figure(figsize=(8, 6))
        plt.savefig(results_dir + f"multiclass_SHAP_{classifiers[0]}.png")
