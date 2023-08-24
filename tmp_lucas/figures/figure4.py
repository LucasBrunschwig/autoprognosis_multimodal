# stdlib
import os

# third party
import matplotlib.pyplot as plt
import psutil
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

# autoprognosis absolute
from autoprognosis.studies.multimodal_classifier import MultimodalStudy
from autoprognosis.utils.serialization import load_model_from_file, save_model_to_file
from autoprognosis.utils.tester import evaluate_multimodal_estimator

from tmp_lucas import DataLoader

if __name__ == "__main__":
    # This script is for late fusion. The goal is to use one model of for image and one for tabular
    # To gain time we predefine the model parameter into multimodal_classifier_combos.py

    bayesian_optimization = True
    explain = True
    fit_model = True
    cross_validation = True

    results_dir = "figure_output/"
    os.makedirs(results_dir, exist_ok=True)

    print("Loading Images")

    print(
        f"GB available before loading data: {psutil.virtual_memory().available/1073741824:.2f}"
    )

    DL = DataLoader(
        path_=r"../../data",
        data_src_="PAD-UFES",
        format_="PIL",
    )

    df_train, df_test = DL.load_dataset(
        raw=False, sample=False, pacheco=False, full_size=False
    )
    group = ["_".join(patient.split("_")[0:2]) for patient in list(df_train.index)]
    df_train["patient"] = group
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    print(
        f"GB available after loading data: {psutil.virtual_memory().available/1073741824:.2f}"
    )

    # Classifier
    classifier = "cnn_fine_tune"
    tabular = "neural_nets"
    predefined_cnn = ["resnet50"]
    study_name = f"image_{classifier}_{predefined_cnn[0]}"
    os.makedirs(f"tmp_image/{study_name}", exist_ok=True)
    study_name = f"late_fusion_{classifier}_{predefined_cnn[0]}_{tabular}_pacheco"

    if bayesian_optimization:
        study = MultimodalStudy(
            study_name=study_name,
            dataset=df_train,  # pandas DataFrame
            multimodal_type="late_fusion",
            image="image",
            target="label",  # the label column in the dataset
            sample_for_search=False,  # no Sampling
            predefined_cnn=predefined_cnn,
            feature_selection=[],
            image_processing=[],
            n_folds_cv=5,
            num_iter=200,
            metric="aucroc",
            classifiers=[tabular],
            image_classifiers=[classifier],
            timeout=int(10 * 3600),
            num_study_iter=1,
            workspace="tmp_late_fusion/",
            random_state=8,
            group_id="patient",
        )

        study.run()

    if explain:

        df_test_label = df_test.label
        df_test_features = df_test.drop(["label"], axis=1)
        df_train_label = df_train.label
        df_train_features = df_train.drop(["label"], axis=1)

        encoder = LabelEncoder().fit(df_train_label)
        df_train_label = encoder.transform(df_train_label)
        df_test_label = encoder.transform(df_test_label)

        # Transform into a dictionary of modalities
        df_features_train_img = df_train_features[["image"]]
        df_features_train_tab = df_train_features.drop(["image"], axis=1)
        df_features_test_img = df_test_features[["image"]]
        df_features_test_tab = df_test_features.drop(["image"], axis=1)
        df_features_train = {"img": df_features_train_img, "tab": df_features_train_tab}
        df_features_test = {"img": df_features_test_img, "tab": df_features_test_tab}

        # Add an explainer and fit the model
        if fit_model:
            model = load_model_from_file("tmp_late_fusion/" + study_name + r"/model.p")
            model.explainer_plugins = ["kernel_shap", "grad_cam"]
            model.fit(df_train_features, df_train_label)
            save_model_to_file(
                "tmp_late_fusion/" + study_name + r"/model_trained.p", model
            )
        else:
            model = load_model_from_file(
                "late_fusion/" + study_name + r"/model_trained.p"
            )

        if cross_validation:
            model_cv = load_model_from_file("tmp_image/" + study_name + r"/model.p")
            results = evaluate_multimodal_estimator(
                model_cv,
                df_train_features,
                df_train_label,
                n_folds=5,
                seed=8,
                group=group,
            )

            print("5-fold CV results, seed = 42")
            for metric, value in results["str"].items():
                print(metric, value)

        predictions = model.predict(df_features_test).astype(int)
        predictions = encoder.inverse_transform(predictions)
        df_test_label = encoder.inverse_transform(df_test_label)
        # Define class names (replace with your actual class names)
        class_names = encoder.classes_

        # Compute confusion matrix
        cm = confusion_matrix(predictions, df_test_label, normalize="true")

        # Create a heatmap plot for the confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )

        # Add labels, title, and axis names
        plt.xlabel("Predicted Diagnoses")
        plt.ylabel("True Diagnoses")
        plt.title("Confusion Matrix for Skin Lesion Classification")
        plt.savefig(
            rf"C:\Users\Lucas\Desktop\Master-Thesis-Cambridge\results\confusion_matrix_{classifier}.png"
        )

        # # Test the explainer on test data
        # results = model.explain_multimodal(df_features_test, df_test_label)
        #
        # plt.figure(figsize=(8, 6))
        # tab_explanation = results["tab"]
        # image_explanation = results["img"]
        #
        # fig, axes = plt.subplots(2, 6, figsize=(36, 12))
        # for explainer, explanation in image_explanation.items():
        #     for label, images in explanation.items():
        #         image = images[0]
        #
        #         axes[0, label].set_title(encoder.classes_[label], fontsize=17)
        #
        #         axes[0, label].imshow(image[0])
        #         axes[0, label].axis("off")
        #
        #         axes[1, label].imshow(image[1])
        #         axes[1, label].axis("off")
        #
        #     axes[0, 0].get_yaxis().set_visible(False)
        #     axes[1, 0].get_yaxis().set_visible(False)
        #
        #     # Add row names to the second row
        #     fig.text(0.06, 0.7, "Original", ha="center", va="center", fontsize=17)
        #     fig.text(0.06, 0.27, "Grad-CAM", ha="center", va="center", fontsize=17)
        #
        #     plt.subplots_adjust(wspace=0.3, hspace=0.3)
        #     plt.savefig(results_dir + "grad_cam_summary.png")
