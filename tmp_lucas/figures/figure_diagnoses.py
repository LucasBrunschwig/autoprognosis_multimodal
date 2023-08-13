# third party
import matplotlib.pyplot as plt
import psutil

from tmp_lucas.loader import DataLoader

if __name__ == "__main__":

    train_model = True
    explain = True

    print("Loading Images")

    print(
        f"GB available before loading data: {psutil.virtual_memory().available/1073741824:.2f}"
    )

    DL = DataLoader(
        path_=r"C:\Users\Lucas\Desktop\Master-Thesis-Cambridge\data",
        data_src_="PAD-UFES",
        format_="PIL",
    )

    df = DL.load_dataset(raw=False, sample=False)

    for i in range(10):

        diagnoses = df.groupby("label").sample(n=1).reset_index(drop=True)
        images = diagnoses["image"].tolist()
        legends = diagnoses["label"].tolist()

        fig, axes = plt.subplots(2, 3, figsize=(20, 10))

        for ax, img, legend in zip(axes.ravel(), images, legends):
            ax.imshow(img)
            ax.axis("off")  # Hide axes
            ax.text(0.5, -0.1, legend, size=20, ha="center", transform=ax.transAxes)

        plt.tight_layout()
        plt.savefig("figure_output/diagnoses" + f"_{i}.png")
