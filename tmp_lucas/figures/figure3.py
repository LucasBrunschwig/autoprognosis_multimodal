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
from autoprognosis.plugins.explainers import Explainers
from autoprognosis.studies.image_classifiers import ImageClassifierStudy
from autoprognosis.utils.serialization import load_model_from_file, save_model_to_file

from tmp_lucas import DataLoader

if __name__ == "__main__":

    train_model = True
    explain = True
    fit_model = True

    results_dir = "figure_output/"
    os.makedirs(results_dir, exist_ok=True)

    print("Loading Images")

    print(
        f"GB available before loading data: {psutil.virtual_memory().available/1073741824:.2f}"
    )

    # Use a subprocess to free memory

    DL = DataLoader(
        path_="../../data",
        data_src_="PAD-UFES",
        format_="PIL",
    )

    df = DL.load_dataset(raw=False, sample=False)

    df = df[["image", "label"]]
    classifier = "cnn_fine_tune"
    predefined_cnn = ["alexnet"]
    study_name = f"image_{classifier}_{predefined_cnn[0]}"  # _{datetime.now().strftime('%Y-%m-%d-%H')}"

    if train_model:
        print("Training", study_name)
        study = ImageClassifierStudy(
            study_name=study_name,
            dataset=df,  # pandas DataFrame
            target="label",  # the label column in the dataset
            sample_for_search=False,  # no Sampling
            predefined_cnn=predefined_cnn,
            n_folds_cv=5,
            num_iter=1,
            metric="aucroc",
            classifiers=[classifier],
            timeout=int(10 * 3600),
            num_study_iter=1,
            workspace="../",
        )

        study.run()

    if explain:

        df.reset_index(drop=True, inplace=True)
        df_label = df.label
        if "hash_image" in df.columns:
            df.drop("hash_image", axis=1, inplace=True)
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

        # Add an explainer and fit the model
        if fit_model:
            model = load_model_from_file("../" + study_name + r"/model.p")
            model.fit(df_features_train, df_label_train)
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
        plt.savefig(results_dir + "confusion_matrix_{classifier}.png")
        plt.figure(figsize=(8, 6))

        # Test the explainer on test data
        explain = Explainers().get(
            "grad_cam",
            model,
            df_features_test,
            df_label_test,
            target_layer="avgpool",
            prefit=True,
            task_type="classification",
        )

        results = explain.explain(df_features_test, df_label_test)
        fig, axes = plt.subplots(2, 6, figsize=(36, 12))
        for label, images in results.items():
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
