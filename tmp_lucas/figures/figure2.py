# stdlib
import os

# third party
import matplotlib.pyplot as plt
import psutil
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# autoprognosis absolute
from autoprognosis.studies.classifiers import ClassifierStudy
from autoprognosis.utils.serialization import load_model_from_file

from tmp_lucas.loader import DataLoader

if __name__ == "__main__":

    train_model = True
    explain = True

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

    df = df[df.columns.difference(["image"])]

    classifier = "logistic_regression"
    study_name = (
        f"tabular_{classifier}_pacheco"  # _{datetime.now().strftime('%Y-%m-%d-%H')}"
    )

    if train_model:
        study = ClassifierStudy(
            study_name=study_name,
            dataset=df,  # pandas DataFrame
            target="label",  # the label column in the dataset
            sample_for_search=False,  # no Sampling
            n_folds_cv=5,
            num_iter=1,
            metric="aucroc",
            classifiers=[classifier],
            timeout=int(10 * 3600),
            num_study_iter=1,
            workspace=r"C:\Users\Lucas\Desktop\Master-Thesis-Cambridge\autoprognosis_multimodal\tmp_lucas",
        )

        study.fit()

    model = load_model_from_file("..\\" + study_name + r"\model.p")

    df.reset_index(drop=True, inplace=True)
    df_label = df.label
    df_features = df.drop(["label"], axis=1)

    # Unique LabelEncoder
    label_encoder = LabelEncoder().fit(df_label)
    df_label = label_encoder.transform(df_label)

    # Prepare test and Training data
    (
        df_features_train,
        df_features_test,
        df_label_train,
        df_label_test,
    ) = train_test_split(
        df_features, df_label, shuffle=True, test_size=0.2, stratify=df_label
    )

    # Add an explainer and fit the model
    model.explainer_plugins = ["kernel_shap"]
    model.fit(df_features_train, df_label_train)
    predictions = model.predict(df_features_test)
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
    plt.figure(figsize=(8, 6))

    # Test the explainer on test data
    model.explain(df_features_test, class_names)
    plt.savefig(results_dir + f"multiclass_SHAP_{classifier}.png")
