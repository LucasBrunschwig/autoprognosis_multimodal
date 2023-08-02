# stdlib
import argparse
import logging
import os
from pathlib import Path
import random
from typing import Union
import zipfile

# third party
from PIL import Image
import cv2
import numpy as np
import pandas as pd
from torchvision import transforms

# autoprognosis absolute
import autoprognosis.logger as log


def shade_of_gray_cc(img, power=6, gamma=None):
    """
    img (numpy array): the original image with format of (h, w, c)
    power (int): the degree of norm, 6 is used in reference paper
    gamma (float): the value of gamma correction, 2.2 is used in reference paper
    """
    img = np.asarray(img)
    img_dtype = img.dtype

    if gamma is not None:
        img = img.astype("uint8")
        look_up_table = np.ones((256, 1), dtype="uint8") * 0
        for i in range(256):
            look_up_table[i][0] = 255 * pow(i / 255, 1 / gamma)
        img = cv2.LUT(img, look_up_table)

    img = img.astype("float32")
    img_power = np.power(img, power)
    rgb_vec = np.power(np.mean(img_power, (0, 1)), 1 / power)
    rgb_norm = np.sqrt(np.sum(np.power(rgb_vec, 2.0)))
    rgb_vec = rgb_vec / rgb_norm
    rgb_vec = 1 / (rgb_vec * np.sqrt(3))
    img = np.multiply(img, rgb_vec)

    # Andrew Anikin suggestion
    img = np.clip(img, a_min=0, a_max=255)

    return Image.fromarray(img.astype(img_dtype))


def crop_center(image, size):
    width, height = image.size
    left = (width - size) // 2
    top = (height - size) // 2
    right = (width + size) // 2
    bottom = (height + size) // 2
    cropped_image = image.crop((left, top, right, bottom))
    return cropped_image


def read_folder(dirpath: Union[Path, str], img_format: str = "PIL") -> dict:
    """Read zipped images where the filename corresponds to the id.

    Args:
        filename: name of the file (.zip)
        img_format: which format would you like to work with (
                    "PIL": PIL.Image.Image, "OpenCV": OpenCV,
                    "Numpy": NumPy.ndarray, "Tensor": PyTorch.Tensor)

    Returns:
        images: dictionary  {id : image}

    """

    images = {}
    files = os.listdir(dirpath)
    try:
        for img_name in files:
            if img_name.endswith(".png"):
                img_ = Image.open(os.path.join(dirpath, img_name))
                img_ = img_.convert("RGB")  # .resize((256, 256))
                img_ = shade_of_gray_cc(img_)
                if img_format.upper() == "NUMPY":
                    img_ = np.asarray(img_)
                elif img_format.upper() == "TENSOR":
                    img_ = transforms.PILToTensor()(img_)
                elif img_format.upper() == "OPENCV":
                    raise NotImplementedError("Future implementations")
                images.update({os.path.basename(img_name): img_})

    except zipfile.BadZipFile:
        log.error("Error: Zip file is corrupted")

    return images


def read_zip(filename: Union[Path, str], img_format: str = "PIL") -> dict:
    """Read zipped images where the filename corresponds to the id.

    Args:
        filename: name of the file (.zip)
        img_format: which format would you like to work with (
                    "PIL": PIL.Image.Image, "OpenCV": OpenCV,
                    "Numpy": NumPy.ndarray, "Tensor": PyTorch.Tensor)

    Returns:
        images: dictionary  {id : image}

    """

    images = {}
    try:
        with zipfile.ZipFile(filename, "r") as file:
            for img_name in file.namelist():
                if img_name.endswith(".png"):
                    try:
                        img_ = Image.open(file.open(img_name))
                        img_ = img_.convert("RGB")
                        if img_format.upper() == "NUMPY":
                            img_ = np.asarray(img_)
                        elif img_format.upper() == "TENSOR":
                            img_ = transforms.PILToTensor()(img_)
                        elif img_format.upper() == "OPENCV":
                            raise NotImplementedError("Future implementations")
                        images.update({os.path.basename(img_name): img_})
                    except ValueError:
                        log.info("empty images")
    except zipfile.BadZipFile:
        log.error("Error: Zip file is corrupted")

    return images


class DataLoader:
    def __init__(self, path_: str, data_src_: str, format_: str):

        # Data Location
        self.PATH = Path(path_)
        self.SOURCE = data_src_
        self.format = format_

        # Data variables
        self.ids = None
        self.metadata = None
        self.images = None
        self.labels = None

    def load_dataset(
        self, split_images=True, classes: list = None, sample: bool = False, raw=False
    ):
        """Returns the loaded dataset based on the source input

        Parameters:
            split_images: bool,
                if true will create one data point per image in the dataset.
            classes: list,
                extract a subset of labels on the dataset.
        Returns:
            3 pd.DataFrame containing images, clinical data, and labels
        """

        # Different loaders for different sources
        if self.SOURCE == "PAD-UFES":
            self._load_pad_ufes_dataset(
                split_images=split_images, classes=classes, sample=sample, raw=raw
            )

        return self.df

    def _load_pad_ufes_dataset(
        self, split_images: bool, classes: list, sample: bool, raw: bool
    ):
        """This loader will load the zipped images and metadata and returns 3 lists

        Args:
            split_images: if true returns one image per data point.

        Returns:
            pandas.DataFrame(id, metadata, image(s), labels)


        Remark:
            - there can be several images for one patient. Some models might integrate multiple images for one
                data point and others treat them separately

        """

        # either drop missing features or drop incomplete rows

        # Load Metadata as DataFrame
        metadata_df = pd.read_csv(self.PATH / "metadata.csv")

        # Load Images as dictionary
        if sample:
            images_dict = read_folder(self.PATH / "imgs_part_1", self.format)
        else:
            images_dict = read_folder(self.PATH / "imgs", self.format)

        images_df = pd.DataFrame.from_dict(
            images_dict, orient="index", columns=["image"], dtype=object
        )
        images_df.reset_index(inplace=True)

        # Merge Metadata and Images
        df = pd.merge(
            images_df, metadata_df, how="left", left_on=["index"], right_on=["img_id"]
        )

        # if multiple images join images on one row
        if not split_images:
            df.groupby(["patient_id"]).agg(
                {
                    **{
                        "image": list,
                    },
                    **{col: "first" for col in df.columns if col != "image"},
                }
            )

        # Reset index
        df["img_id"] = df["img_id"].apply(lambda t: t[:-4])
        df.set_index("img_id", inplace=True)
        df.drop(
            columns=["patient_id", "lesion_id", "index", "biopsed"],
            axis=1,
            inplace=True,
        )
        df.rename(columns={"diagnostic": "label"}, inplace=True)

        # Remove features only present in cancerous patients

        # Transform df into suitable numeric values
        one_hot_encoding = False
        drop_features = True
        drop_rows = False

        if not raw:
            if one_hot_encoding:
                categorical_var = [
                    "background_father",
                    "background_mother",
                    "gender",
                    "region",
                ]
                for col in df.columns:
                    if col in categorical_var:
                        tmp_one_hot = pd.get_dummies(df[col], prefix=col)
                        df.drop(col, axis=1, inplace=True)
                        df = df.join(tmp_one_hot)

                # Remove Unknown and convert False True to integers
                df.replace("UNK", np.nan, inplace=True)
                df_images = df.image.to_frame()
                df.drop("image", axis=1, inplace=True)
                df.replace({False: 0, True: 1, "False": 0, "True": 0}, inplace=True)
                df = df_images.join(df)

            # Pacheto paper
            elif drop_features:
                incomplete_features = []
                for column in df.columns:
                    if sum(df[column].isna()) and column not in [
                        "diameter_1",
                        "diameter_2",
                    ]:
                        incomplete_features.append(column)
                df.drop(incomplete_features, axis=1, inplace=True)

                categorical_features = [
                    "region",
                    "itch",
                    "grew",
                    "hurt",
                    "changed",
                    "bleed",
                    "elevation",
                ]
                for column in categorical_features:
                    tmp = pd.get_dummies(df[column], prefix=column)
                    if column + "_UNK" in tmp.columns:
                        tmp.drop([column + "_UNK"], axis=1, inplace=True)
                    df.drop(column, axis=1, inplace=True)
                    df = df.join(tmp)

            # drop incomplete rows
            elif drop_rows:
                for ix, row in df.iterrows():
                    pass

        if classes is not None:
            df = df[df["label"].isin(classes)]

        # Get one df per source
        self.ids = df.index
        self.df = df
        self.labels = df.label.to_frame()

    def summary(self):
        logging.info(f"Number of data points: {len(self.metadata)}")
        logging.info(
            f"Number of unique patients: {np.unique(self.metadata.patient_id.to_numpy())}"
        )
        logging.info(f"Number of diagnoses: {self.labels.value_counts()}")

    def get_training_validation_test_dataset(self):
        pass

    def sample_dataset(self, n):
        """Returns a sample of the dataset

        Args:
            n: int, the number of data points returned
        """

        indices = random.sample(range(0, len(self.df)), n)

        return self.df.iloc[indices, :]


if __name__ == "__main__":

    # Create a temporary parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        help="provide the directory where dataset is stored",
        type=str,
        default="",
    )
    parser.add_argument("--data_src", help="provide the data source type")
    args = parser.parse_args()

    data_src = args.data_src
    input_dir = args.input_dir

    DL = DataLoader(input_dir, data_src)
    DL.load_dataset()

    # DL.summary()
