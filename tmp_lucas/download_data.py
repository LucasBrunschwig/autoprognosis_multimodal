# stdlib
import os
import zipfile

# third party
import gdown

if __name__ == "__main__":
    os.makedirs("../data", exist_ok=True)

    # download imagazes
    url_imgs = "https://drive.google.com/file/d/1XIO-PMqi0ZlrkGRYP-oWwNLcDgR73Quk/view?usp=sharing"
    output = "../data/imgs.zip"
    gdown.download(url_imgs, output, quiet=False, fuzzy=True)
    with zipfile.ZipFile(output, "r") as zip_ref:
        zip_ref.extractall("../data")

    url_metadata = "https://drive.google.com/file/d/1wPiXDcdijb0rSECQwIqAiHUGlizvywbP/view?usp=sharing"
    output = "../data/metadata.csv"
    gdown.download(url_metadata, output, quiet=False, fuzzy=True)
