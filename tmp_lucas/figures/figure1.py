# stdlib
import os

# third party
import matplotlib.pyplot as plt
import pandas as pd
import psutil
from sklearn.manifold import TSNE

# autoprognosis absolute
from autoprognosis.plugins.imputers import Imputers
from autoprognosis.plugins.preprocessors import Preprocessors

from tmp_lucas.loader import DataLoader

if __name__ == "__main__":

    train_model = True
    explain = True

    print("Loading Images")

    print(
        f"GB available before loading data: {psutil.virtual_memory().available/1073741824:.2f}"
    )

    results_dir = "figure_output/"
    os.makedirs(results_dir, exist_ok=True)

    DL = DataLoader(
        path_=r"../../data",
        data_src_="PAD-UFES",
        format_="PIL",
    )

    df = DL.load_dataset(raw=False, sample=False)

    df = df[df.columns.difference(["image"])]
    variables = [
        "age",
        "elevation",
        "hurt",
        "itch",
        "grew",
        "bleed",
        "region",
        "diameter_1",
        "diameter_2",
    ]
    label = ["label"]
    numerical_variable = ["age", "diameter_1", "diameter_2"]
    categorical_variable = list(set(variables) - set(numerical_variable))

    categorical_columns = []
    column_name = []
    for col in df.columns:
        for variable in variables:
            if variable in col:
                if variable in categorical_variable:
                    categorical_columns.append(col)
                column_name.append(col)
                break
    df_label = df[label].copy()
    df = df[column_name]

    imputers = Imputers().get("mice")
    df = imputers.fit_transform(df)
    df.columns = column_name

    # Standard Scaler for numerical variables
    standard_scaling = Preprocessors(category="feature_scaling").get("scaler")
    df_standard = standard_scaling.fit_transform(df[numerical_variable])
    df_standard.columns = df[numerical_variable].columns
    df[numerical_variable] = df_standard[numerical_variable]

    # PCA on all the data points
    feature_reduction = Preprocessors(category="dimensionality_reduction").get("pca")
    # df_reduced = feature_reduction.fit_transform(df)
    df_reduced = pd.DataFrame(
        TSNE(n_components=2, learning_rate="auto", init="random").fit_transform(df)
    )
    df_reduced[label] = df_label.reset_index(drop=True)

    colors_points = {
        "ACK": "red",
        "BCC": "blue",
        "MEL": "orange",
        "SCC": "green",
        "SEK": "violet",
        "NEV": "grey",
    }
    plt.figure(figsize=(8, 6))

    for ix, row in df_reduced.iterrows():
        x = row.iloc[0]
        y = row.iloc[1]
        label_ = row.loc["label"]
        plt.scatter(x, y, c=colors_points[label_], s=10, alpha=0.3)

    group_centroids = df_reduced.groupby(label).mean()
    for ix, centroids in group_centroids.iterrows():
        x = centroids.iloc[0]
        y = centroids.iloc[1]
        plt.scatter(x, y, c=colors_points[ix], s=100, alpha=1.0, label=ix)

    plt.xlabel("t-sne 1")
    plt.ylabel("t-sne 2")
    plt.title("T-SNE - Skin Lesions, Clinical Features")
    plt.legend()
    plt.savefig(results_dir + "tabular_tsne.png")

    # # Extract Centroids
    # group_centroids = df.groupby(label).agg({col: "mean" for col in categorical_columns})
    # group_centroids = pd.concat([df.groupby(label)[numerical_variable].mean(), group_centroids], axis=1)
    #
    # pca_results = feature_reduction.fit_transform(group_centroids)
    # pca_results.index = df.label.value_counts().index
    #
    #
    # for label, row in pca_results.iterrows():
    #     x = row.iloc[0]
    #     y = row.iloc[1]
    #     plt.scatter(x, y, label=label, s=100)
    #
