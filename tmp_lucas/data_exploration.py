# third party
from loader import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# autoprognosis absolute
from autoprognosis.plugins.preprocessors import Preprocessors

# SETTINGS

SAVE_RESULTS = r"C:\Users\Lucas\Desktop\Master-Thesis-Cambridge\results"
EXPLORATORY_ANALYSIS = True
CORRELATION_IMAGES = False

# DATA LOADER

DL = DataLoader(
    path_=r"C:\Users\Lucas\Desktop\Master-Thesis-Cambridge\data",
    data_src_="PAD-UFES",
    format_="PIL",
)


def cramers_v(x, y):
    """Compute Cramér's V statistic for categorial-categorial association."""
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def exploratory_analysis():
    df = DL.load_dataset(raw=True)

    for column in df.columns.difference(["image"]):
        df[column].replace("UNK", np.nan, inplace=True)

    correlation_known = []
    labels_known = []
    for column in df.columns.difference(["image", "label"]):
        if sum(df[column].isna()):
            print(column, "unknown")
            # print(df[["label", column]][df[column].isna()]["label"].value_counts())
            feature = df[~df[column].isna()][column]
            if column in ["diameter_1", "diameter_2"]:
                feature = pd.cut(df[column], bins=20, labels=range(20))
            correlation_known.append(cramers_v(feature, df["label"]))
            labels_known.append(column.replace("_", " "))
        else:
            print(column, "known")
            feature = df[column]
            if column == "age":
                feature = pd.cut(df[column], bins=30, labels=range(30))
            correlation_known.append(cramers_v(feature, df["label"]))
            labels_known.append(column.replace("_", " "))

    # Sort values in descending order and get the indices
    sorted_indices = sorted(
        range(len(correlation_known)), key=lambda k: correlation_known[k], reverse=True
    )

    # Rearrange values and labels based on the sorted indices
    sorted_values = [correlation_known[i] for i in sorted_indices]
    sorted_labels = [labels_known[i] for i in sorted_indices]

    sns.set_theme()

    # Plot
    plt.figure(figsize=(10, 10))
    plt.bar(
        sorted_labels,
        sorted_values,
        width=0.5,
    )
    plt.xlabel("Labels", fontsize=14)
    plt.xticks(rotation=90, fontsize=14)  # Rotate x-axis labels by 90 degrees
    plt.ylabel("Cramer Coefficient", fontsize=14)
    plt.title("Cramer Coefficient for metadata and diagnoses", fontsize=14)
    plt.ylim(0, 1.0)
    plt.grid()
    plt.savefig(SAVE_RESULTS + r"\cramer.png", bbox_inches="tight")

    df_image = df["image"]
    pixels_ = []
    for image in df_image:
        pixels_.append(image.size[0])

    print("min resolution", np.min(pixels_))
    print("max resolution", np.max(pixels_))

    plt.figure()
    plt.hist(pixels_, bins=50)
    plt.ylabel("counts")
    plt.xlabel("resolution (n x n)")
    plt.savefig(SAVE_RESULTS + r"\image_resolution.png", bbox_inches="tight")

    labels = df.label.value_counts().index.tolist()

    labels_df = []
    for label in labels:
        labels_df.append(df[df.label == label])

    for i, label in enumerate(labels):
        labels[i] = labels[i] + "(" + str(df.label.value_counts()[label]) + ")"

    columns = df.columns.difference(["label"]).tolist()

    na_perc = np.empty((len(columns), len(labels)))

    for i, column in enumerate(columns):
        for j, label_df in enumerate(labels_df):
            na_perc[i, j] = sum(label_df[column].isna()) / len(label_df)

    # One plot with additional features presence
    na_perc_df = pd.DataFrame(na_perc, index=columns, columns=labels)
    plt.figure()
    sns.heatmap(na_perc_df, cmap="coolwarm", annot=True, cbar=False)
    plt.savefig(SAVE_RESULTS + r"\missing_features.png", bbox_inches="tight")

    # One plot for correlation between diagnosis and region parts
    na_perc = np.round(na_perc, decimals=3)
    regions = np.array(df.region.value_counts().index.tolist())
    region_perc = np.empty((len(regions), len(labels)))
    for j, label_df in enumerate(labels_df):
        for idx, row in label_df.iterrows():
            i = np.argwhere(row["region"] == regions)[0][0]
            region_perc[i, j] += 1 / len(label_df)

    region_perc = np.round(region_perc, decimals=3)
    region_perc_df = pd.DataFrame(region_perc, index=regions, columns=labels)
    plt.figure()
    sns.heatmap(region_perc_df, cmap="coolwarm", annot=True, cbar=False)
    plt.savefig(SAVE_RESULTS + r"\regions_vs_diagnoses.png", bbox_inches="tight")

    # Correlation between biopsied and classes
    # One plot for correlation between diagnosis and region parts
    biopsied = np.array(df.biopsed.value_counts().index.tolist())

    biopsied_perc = np.empty((len(biopsied), len(labels)))
    for j, label_df in enumerate(labels_df):
        for idx, row in label_df.iterrows():
            i = np.argwhere(row["biopsed"] == biopsied)[0][0]
            biopsied_perc[i, j] += 1 / len(label_df)

    biopsied_perc = np.round(biopsied_perc, decimals=3)
    biopsied_perc_df = pd.DataFrame(biopsied_perc, index=biopsied, columns=labels)
    plt.figure()
    sns.heatmap(biopsied_perc_df, cmap="coolwarm", annot=True, cbar=False)
    plt.savefig(SAVE_RESULTS + r"\biopsied_vs_diagnoses.png", bbox_inches="tight")


def correlation_images(df):
    df_list = []
    labels = df.label.value_counts().index.tolist()
    for label in labels:
        df_list.append(
            df[df.label == label][["image"]].sample(
                n=min(len(df[df.label == label]), 50)
            )
        )

    num_groups = len(df_list)
    inter_correlations = np.zeros((num_groups, num_groups))

    for i in range(num_groups):
        group1 = df_list[i]
        intra_correlations = []
        # intra correlation
        for m in range(len(group1)):
            img1 = group1.iloc[m]["image"]
            arr1 = np.array(img1).flatten()
            for n in range(m + 1, len(group1)):
                img2 = group1.iloc[n]["image"]
                arr2 = np.array(img2).flatten()
                intra_correlations.append(np.corrcoef(arr1, arr2)[0, 1])
        inter_correlations[i][i] = np.mean(intra_correlations)

        for j in range(i + 1, num_groups):

            # Intergroup correlation
            group2 = df_list[j]

            corr_sum = []
            for k, img1 in enumerate(group1.image.tolist()):
                arr1 = np.array(img1).flatten()
                intra_correlations.append([])

                for img2 in group2.image:

                    arr2 = np.array(img2).flatten()
                    corr = np.corrcoef(arr1, arr2)[0, 1]
                    corr_sum.append(corr)

            average_corr = np.mean(corr_sum)
            inter_correlations[i][j] = average_corr

    correlation_df = pd.DataFrame(inter_correlations, index=labels, columns=labels)
    plt.figure()
    sns.heatmap(correlation_df, cmap="coolwarm", annot=True, cbar=False)
    plt.savefig(SAVE_RESULTS + r"\class_image_correlations.png")

    return 1


def correlation_cnn_imagenet(df):
    cnn_imagenet = Preprocessors(category="image_reduction").get("cnn_imagenet")

    labels = df.label.value_counts().index.tolist()

    df_list = []
    label_list = []
    for label in labels:
        df_label = df[df.label == label].sample(n=min(len(df[df.label == label]), 50))
        label_list.append(df_label.label)
        df_label.drop(["label"], axis=1, inplace=True)
        df_list.append(df_label[["image"]])

    label_extraction_features = []

    for df, label in zip(df_list, label_list):
        cnn_imagenet.fit(df, label)
        result = cnn_imagenet.transform(df)
        label_extraction_features.append([result])

    label_extraction_features = np.asarray(label_extraction_features).squeeze()

    num_groups = label_extraction_features.shape[0]

    inter_correlations = np.zeros((num_groups, num_groups))

    for i in range(num_groups):
        group1 = label_extraction_features[i]
        intra_correlations = []

        # intra correlation
        for m in range(len(group1)):
            for n in range(m + 1, len(group1)):
                intra_correlations.append(np.corrcoef(group1[m], group1[n])[0, 1])
        inter_correlations[i][i] = np.mean(intra_correlations)

        # Intergroup correlation
        for j in range(i + 1, num_groups):
            group2 = label_extraction_features[j]

            corr_sum = []
            for feature1 in group1:
                intra_correlations.append([])
                for feature2 in group2:

                    corr = np.corrcoef(feature1, feature2)[0, 1]
                    corr_sum.append(corr)

            average_corr = np.mean(corr_sum)
            inter_correlations[i][j] = average_corr

    correlation_df = pd.DataFrame(inter_correlations, index=labels, columns=labels)
    plt.figure()
    sns.heatmap(correlation_df, cmap="coolwarm", annot=True, cbar=False)
    plt.savefig(SAVE_RESULTS + r"\cnn_feature_extraction_correlations.png")


def correlation_cnn_fine_tune(df):
    cnn_fine_tune = Preprocessors(category="image_reduction").get("cnn_fine_tune")
    cnn_fine_tune.lr = 1e-5
    cnn_fine_tune.conv_net = "resnet50"
    cnn_fine_tune.n_hidden_units = 128
    cnn_fine_tune.non_linear = "selu"
    cnn_fine_tune.n_unfrozen_layer = 1

    labels = df.label.value_counts().index.tolist()

    df_list = []
    label_list = []
    for label in labels:
        df_label = df[df.label == label].sample(n=min(len(df[df.label == label]), 50))
        label_list.append(df_label.label)
        df_label.drop(["label"], axis=1, inplace=True)
        df_list.append(df_label[["image"]])

    df_label = df["label"]
    df_label = pd.DataFrame(LabelEncoder().fit_transform(df_label))
    df.drop(["label"], axis=1, inplace=True)
    cnn_fine_tune.fit(df[["image"]], df_label)

    label_extraction_features = []

    for df, label in zip(df_list, label_list):
        result = cnn_fine_tune.transform(df)
        label_extraction_features.append([result])

    label_extraction_features = np.asarray(label_extraction_features).squeeze()

    num_groups = label_extraction_features.shape[0]

    inter_correlations = np.zeros((num_groups, num_groups))

    for i in range(num_groups):
        group1 = label_extraction_features[i]
        intra_correlations = []

        # intra correlation
        for m in range(len(group1)):
            for n in range(m + 1, len(group1)):
                intra_correlations.append(np.corrcoef(group1[m], group1[n])[0, 1])
        inter_correlations[i][i] = np.mean(intra_correlations)

        # Intergroup correlation
        for j in range(i + 1, num_groups):
            group2 = label_extraction_features[j]

            corr_sum = []
            for feature1 in group1:
                intra_correlations.append([])
                for feature2 in group2:

                    corr = np.corrcoef(feature1, feature2)[0, 1]
                    corr_sum.append(corr)

            average_corr = np.mean(corr_sum)
            inter_correlations[i][j] = average_corr

    correlation_df = pd.DataFrame(inter_correlations, index=labels, columns=labels)
    plt.figure()
    sns.heatmap(correlation_df, cmap="coolwarm", annot=True, cbar=False)
    plt.savefig(SAVE_RESULTS + r"\cnn_fine_tune_feature_extraction_correlations.png")


def correlation_cnn(df):
    cnn_fine_tune = Preprocessors(category="image_reduction").get("cnn")
    cnn_fine_tune.lr = 1e-5
    cnn_fine_tune.conv_net = "resnet50"
    labels = df.label.value_counts().index.tolist()

    resizer = Preprocessors(category="image_processing").get("resizer")
    df["image"] = resizer.fit_transform(df[["image"]]).squeeze().values
    normalizer = Preprocessors(category="image_processing").get("normalizer")
    df["image"] = normalizer.fit_transform(df[["image"]]).squeeze().values

    df_list = []
    label_list = []
    for label in labels:
        df_label = df[df.label == label].sample(n=min(len(df[df.label == label]), 50))
        label_list.append(df_label.label)
        df_label.drop(["label"], axis=1, inplace=True)
        df_list.append(df_label[["image"]])

    df_label = df["label"]
    df_label = pd.DataFrame(LabelEncoder().fit_transform(df_label))

    cnn_fine_tune.fit(df[["image"]], df_label)

    label_extraction_features = []

    for df, label in zip(df_list, label_list):
        result = cnn_fine_tune.transform(df)
        label_extraction_features.append([result])

    label_extraction_features = np.asarray(label_extraction_features).squeeze()

    num_groups = label_extraction_features.shape[0]

    inter_correlations = np.zeros((num_groups, num_groups))

    for i in range(num_groups):
        group1 = label_extraction_features[i]
        intra_correlations = []

        # intra correlation
        for m in range(len(group1)):
            for n in range(m + 1, len(group1)):
                intra_correlations.append(np.corrcoef(group1[m], group1[n])[0, 1])
        inter_correlations[i][i] = np.mean(intra_correlations)

        # Intergroup correlation
        for j in range(i + 1, num_groups):
            group2 = label_extraction_features[j]

            corr_sum = []
            for feature1 in group1:
                intra_correlations.append([])
                for feature2 in group2:

                    corr = np.corrcoef(feature1, feature2)[0, 1]
                    corr_sum.append(corr)

            average_corr = np.mean(corr_sum)
            inter_correlations[i][j] = average_corr

    correlation_df = pd.DataFrame(inter_correlations, index=labels, columns=labels)
    plt.figure()
    sns.heatmap(correlation_df, cmap="coolwarm", annot=True, cbar=False)
    plt.savefig(SAVE_RESULTS + r"\cnn_scratch_feature_extraction_correlations.png")


def correlation_simsiam(df):
    cnn_simsiam = Preprocessors(category="image_reduction").get("simsiam")
    labels = df.label.value_counts().index.tolist()

    resizer = Preprocessors(category="image_processing").get("resizer")
    df["image"] = resizer.fit_transform(df[["image"]]).squeeze().values
    normalizer = Preprocessors(category="image_processing").get("normalizer")
    df["image"] = normalizer.fit_transform(df[["image"]]).squeeze().values

    df_list = []
    label_list = []
    for label in labels:
        df_label = df[df.label == label].sample(n=min(len(df[df.label == label]), 50))
        label_list.append(df_label.label)
        df_label.drop(["label"], axis=1, inplace=True)
        df_list.append(df_label[["image"]])

    df_label = df["label"]
    df_label = pd.DataFrame(LabelEncoder().fit_transform(df_label))
    cnn_simsiam.fit(df[["image"]], df_label)

    label_extraction_features = []

    for df, label in zip(df_list, label_list):
        result = cnn_simsiam.transform(df)
        label_extraction_features.append([result])

    label_extraction_features = np.asarray(label_extraction_features).squeeze()

    num_groups = label_extraction_features.shape[0]

    inter_correlations = np.zeros((num_groups, num_groups))

    for i in range(num_groups):
        group1 = label_extraction_features[i]
        intra_correlations = []

        # intra correlation
        for m in range(len(group1)):
            for n in range(m + 1, len(group1)):
                intra_correlations.append(np.corrcoef(group1[m], group1[n])[0, 1])
        inter_correlations[i][i] = np.mean(intra_correlations)

        # Intergroup correlation
        for j in range(i + 1, num_groups):
            group2 = label_extraction_features[j]

            corr_sum = []
            for feature1 in group1:
                intra_correlations.append([])
                for feature2 in group2:

                    corr = np.corrcoef(feature1, feature2)[0, 1]
                    corr_sum.append(corr)

            average_corr = np.mean(corr_sum)
            inter_correlations[i][j] = average_corr

    correlation_df = pd.DataFrame(inter_correlations, index=labels, columns=labels)
    plt.figure()
    sns.heatmap(correlation_df, cmap="coolwarm", annot=True, cbar=False)
    plt.savefig(SAVE_RESULTS + r"\cnn_scratch_feature_extraction_correlations.png")


def correlation_contrastive_learning(df):
    cnn_simsiam = Preprocessors(category="image_reduction").get("contrastive_learning")
    labels = df.label.value_counts().index.tolist()

    resizer = Preprocessors(category="image_processing").get("resizer")
    df["image"] = resizer.fit_transform(df[["image"]]).squeeze().values
    normalizer = Preprocessors(category="image_processing").get("normalizer")
    df["image"] = normalizer.fit_transform(df[["image"]]).squeeze().values

    df_list = []
    label_list = []
    for label in labels:
        df_label = df[df.label == label].sample(n=min(len(df[df.label == label]), 50))
        label_list.append(df_label.label)
        df_label.drop(["label"], axis=1, inplace=True)
        df_list.append(df_label[["image"]])

    df_label = df["label"]
    df_label = pd.DataFrame(LabelEncoder().fit_transform(df_label))
    cnn_simsiam.fit(df[["image"]], df_label)

    label_extraction_features = []

    for df, label in zip(df_list, label_list):
        result = cnn_simsiam.transform(df)
        label_extraction_features.append([result])

    label_extraction_features = np.asarray(label_extraction_features).squeeze()

    num_groups = label_extraction_features.shape[0]

    inter_correlations = np.zeros((num_groups, num_groups))

    for i in range(num_groups):
        group1 = label_extraction_features[i]
        intra_correlations = []

        # intra correlation
        for m in range(len(group1)):
            for n in range(m + 1, len(group1)):
                intra_correlations.append(np.corrcoef(group1[m], group1[n])[0, 1])
        inter_correlations[i][i] = np.mean(intra_correlations)

        # Intergroup correlation
        for j in range(i + 1, num_groups):
            group2 = label_extraction_features[j]

            corr_sum = []
            for feature1 in group1:
                intra_correlations.append([])
                for feature2 in group2:

                    corr = np.corrcoef(feature1, feature2)[0, 1]
                    corr_sum.append(corr)

            average_corr = np.mean(corr_sum)
            inter_correlations[i][j] = average_corr

    correlation_df = pd.DataFrame(inter_correlations, index=labels, columns=labels)
    plt.figure()
    sns.heatmap(correlation_df, cmap="coolwarm", annot=True, cbar=False)
    plt.savefig(SAVE_RESULTS + r"\cnn_scratch_feature_extraction_correlations.png")


if __name__ == "__main__":

    if EXPLORATORY_ANALYSIS:
        exploratory_analysis()

    if CORRELATION_IMAGES:
        df = DL.load_dataset(raw=True)
        # correlation_images(df)

        # TBD
        # correlation_cnn()
        # correlation_cnn_fine_tune()
        # correlation_cnn_imagenet(df)
        # correlation_cnn_fine_tune(df)
        # correlation_cnn(df)
        # correlation_simsiam(df)
        correlation_contrastive_learning(df)
