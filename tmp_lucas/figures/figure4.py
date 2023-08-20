# stdlib
import os

# third party
import matplotlib.pyplot as plt
import pandas as pd
import psutil
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# autoprognosis absolute
from autoprognosis.studies.multimodal_classifier import MultimodalStudy
from autoprognosis.utils.serialization import load_model_from_file, save_model_to_file

from tmp_lucas import DataLoader

if __name__ == "__main__":
    # This script is for late fusion
    train_model = True
    explain = True
    fit_model = True

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

    df = DL.load_dataset(raw=False, sample=False, pacheco=True, full_size=False)

    print(
        f"GB available after loading data: {psutil.virtual_memory().available / 1073741824:.2f}"
    )

    classifier = "cnn_fine_tune"
    predefined_cnn = ["alexnet"]
    tabular = "neural_nets"
    study_name = f"late_fusion_{classifier}_{predefined_cnn[0]}_{tabular}_pacheco"

    if train_model:
        study = MultimodalStudy(
            study_name=study_name,
            dataset=df,  # pandas DataFrame
            multimodal_type="late_fusion",
            image="image",
            target="label",  # the label column in the dataset
            sample_for_search=False,  # no Sampling
            predefined_cnn=predefined_cnn,
            feature_selection=[],
            image_processing=[],
            n_folds_cv=5,
            num_iter=50,
            metric="aucroc",
            classifiers=[tabular],
            image_classifiers=[classifier],
            timeout=int(6 * 3600),
            num_study_iter=1,
            workspace="../",
        )

        study.run()

    if explain:

        df.reset_index(drop=True, inplace=True)
        df_label = df.label
        df_features = df.drop(["label"], axis=1)

        # Unique LabelEncoder
        label_encoder = LabelEncoder().fit(df_label)
        df_label = pd.DataFrame(label_encoder.transform(df_label))

        # Prepare test and Training data
        (
            df_features_train,
            df_features_test,
            df_label_train,
            df_label_test,
        ) = train_test_split(
            df_features,
            df_label,
            shuffle=True,
            test_size=0.2,
            stratify=df_label.squeeze(),
        )

        # Transform into a dictionary of modalities
        df_features_train_img = df_features_train[["image"]]
        df_features_train_tab = df_features_train.drop(["image"], axis=1)
        df_features_test_img = df_features_test[["image"]]
        df_features_test_tab = df_features_test.drop(["image"], axis=1)

        df_features_train = {"img": df_features_train_img, "tab": df_features_train_tab}
        df_features_test = {"img": df_features_test_img, "tab": df_features_test_tab}

        # Add an explainer and fit the model
        if fit_model:
            model = load_model_from_file("../" + study_name + r"/model.p")
            model.explainer_plugins = ["kernel_shap", "grad_cam"]
            model.fit(df_features_train.copy(), df_label_train)
            save_model_to_file("../" + study_name + r"/model_trained.p", model)
        else:
            model = load_model_from_file("../" + study_name + r"/model_trained.p")

        predictions = model.predict(df_features_test).astype(int)
        predictions = label_encoder.inverse_transform(predictions)
        df_label_test = label_encoder.inverse_transform(df_label_test)
        # Define class names (replace with your actual class names)
        class_names = label_encoder.classes_

        # Compute confusion matrix
        cm = confusion_matrix(predictions, df_label_test)

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

        # Test the explainer on test data
        results = model.explain_multimodal(df_features_test, df_label_test)

        plt.figure(figsize=(8, 6))
        tab_explanation = results["tab"]
        image_explanation = results["img"]

        fig, axes = plt.subplots(2, 6, figsize=(36, 12))
        for explainer, explanation in image_explanation.items():
            for label, images in explanation.items():
                image = images[0]

                axes[0, label].set_title(label_encoder.classes_[label], fontsize=17)

                axes[0, label].imshow(image[0])
                axes[0, label].axis("off")

                axes[1, label].imshow(image[1])
                axes[1, label].axis("off")

            axes[0, 0].get_yaxis().set_visible(False)
            axes[1, 0].get_yaxis().set_visible(False)

            # Add row names to the second row
            fig.text(0.06, 0.7, "Original", ha="center", va="center", fontsize=17)
            fig.text(0.06, 0.27, "Grad-CAM", ha="center", va="center", fontsize=17)

            plt.subplots_adjust(wspace=0.3, hspace=0.3)
            plt.savefig(results_dir + "grad_cam_summary.png")
