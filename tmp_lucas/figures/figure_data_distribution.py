# stdlib
import os

# third party
import matplotlib.pyplot as plt
import psutil
import seaborn as sns

from tmp_lucas.loader import DataLoader

if __name__ == "__main__":
    print("Loading Images")

    print(
        f"GB available before loading data: {psutil.virtual_memory().available / 1073741824:.2f}"
    )

    results_dir = "figure_output/"
    os.makedirs(results_dir, exist_ok=True)

    DL = DataLoader(
        path_=r"../../data",
        data_src_="PAD-UFES",
        format_="PIL",
    )

    df = DL.load_dataset(raw=True, sample=False)
    df = df.replace({"True": "YES", "False": "NO", True: "YES", False: "NO"})

    # Age Boxplot
    sns.boxplot(y="age", x="label", data=df[["age", "label"]], palette="Blues_d")
    plt.grid(color="black", linestyle="dotted", linewidth=0.7)
    plt.xlabel("Skin Lesion Diagnoses")
    plt.ylabel("Age")
    plt.savefig("figure_output/age_boxplot.png", dpi=200)

    # Skin Region

    # Evolution Features
    def plot_count(_feat):
        sub_data = df[[_feat, "label"]]
        g = sns.FacetGrid(sub_data, col="label")
        g.map(sns.countplot, _feat, order=["YES", "NO", "UNK"], palette="Blues_d")
        g.savefig(f"figure_output/count_{_feat}.png", dpi=200)

    evolution = ["itch", "grew", "changed", "elevation", "bleed", "hurt"]
    for _feat in evolution:
        plot_count(_feat)
