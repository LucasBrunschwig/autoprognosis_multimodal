from typing import Union

import argparse
import os
import pandas as pd
import zipfile
import logging
from pathlib import Path
from PIL import Image
import numpy as np

logging.basicConfig(format='%(asctime)s | %(levelname)s - %(message)s', level=logging.INFO, datefmt='%m/%d/%Y %I:%M:%S')


def read_zip(filename: Union[Path, str]) -> dict:
    """ Read zipped images where the filename corresponds to the id.

        Args:
            filename: name of the file (.zip)

        Returns:
            images: dictionary  {id : image}

    """

    images = {}
    try:
        with zipfile.ZipFile(filename, 'r') as file:
            for img_name in file.namelist():
                if img_name.endswith(".png"):
                    try:
                        img_ = Image.open(file.open(img_name))
                        images.update({os.path.basename(img_name): np.asarray(img_)})
                    except ValueError:
                        logging.info("empty images")
    except zipfile.BadZipFile:
        logging.error('Error: Zip file is corrupted')

    return images


class DataLoader:
    def __init__(self,
                 path_: str,
                 data_src_: str):

        # Data Location
        self.PATH = Path(path_)
        self.SOURCE = data_src_

        # Data variables
        self.ids = None
        self.metadata = None
        self.images = None
        self.labels = None

    def load_dataset(self):
        """ Returns the loaded dataset based on the source input """

        # Different loaders for different sources
        if self.SOURCE == "PAD-UFES":
            self._load_pad_ufes_dataset()

        return [self.ids, self.images, self.metadata, self.labels]

    def _load_pad_ufes_dataset(self, split_images: bool = False) -> list:
        """ This loader will load the zipped images and metadata and returns 3 lists

            Args:
                split_images: if true returns one image per data point.

            Returns:
                pandas.DataFrame(id, metadata, image(s), labels)


            Remark:
                - there can be several images for one patient. Some models might integrate multiple images for one
                    data point and others treat them separately

        """

        # Load Metadata as DataFrame
        metadata_df = pd.read_csv(self.PATH / "metadata.csv")

        # Load Images as DataFrame
        images_part1 = read_zip(self.PATH / "imgs_part_1.zip")
        images_part2 = read_zip(self.PATH / "imgs_part_2.zip")
        images_part3 = read_zip(self.PATH / "imgs_part_3.zip")
        images = {**images_part1, **images_part2, **images_part3}
        images_df = pd.DataFrame.from_dict(images, orient="index", columns=['images'])
        images_df.reset_index(inplace=True)

        # Merge Metadata and Images
        df = pd.merge(images_df, metadata_df, how='left', left_on=['index'], right_on=['img_id'])

        # if multiple images join images on one row
        if not split_images:
            df.groupby(['patient_id']).agg({**{'images': list, },
                                            **{col: 'first' for col in df.columns if col != 'images'}})

        # Reset index
        df['img_id'] = df['img_id'].apply(lambda t: t[:-4])
        df.set_index('img_id', inplace=True)
        df.drop(columns=['patient_id', 'lesion_id', 'index'], inplace=True)
        df.rename(columns={'diagnostic': 'label'}, inplace=True)

        # Transform df into suitable numeric values

        categorical_var = ['background_father', 'background_mother', 'gender', 'region']
        for col in df.columns:
            if col in categorical_var:
                tmp_one_hot = pd.get_dummies(df[col], prefix=col)
                df.drop(col, axis=1, inplace=True)
                df = df.join(tmp_one_hot)

        # Remove Unknown and convert False True to integers
        df.replace('UNK', np.nan, inplace=True)
        df_images = df.images.to_frame()
        df.drop('images', axis=1, inplace=True)
        df.replace({False: 0, True: 1, 'False': 0, 'True': 0}, inplace=True)
        df = df_images.join(df)

        # Get one df per source
        self.ids = df.index
        self.images = df.images.to_frame()
        self.metadata = df[df.columns[~df.columns.isin(['images', 'label'])]]
        self.labels = df.label.to_frame()

    def summary(self):
        logging.info(f"Number of data points: {len(self.metadata)}")
        logging.info(f"Number of unique patients: {np.unique(self.metadata.patient_id.to_numpy())}")
        logging.info(f"Number of diagnoses: {self.labels.value_counts()}")

    def get_training_validation_test_dataset(self):
        pass

    def sample_dataset(self):
        pass


if __name__ == "__main__":

    # Create a temporary parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",
                        help="provide the directory where dataset is stored",
                        type=str,
                        default="")
    parser.add_argument("--data_src",
                        help="provide the data source type")
    args = parser.parse_args()

    data_src = args.data_src
    input_dir = args.input_dir

    DL = DataLoader(input_dir, data_src)
    DL.load_dataset()

    #DL.summary()
