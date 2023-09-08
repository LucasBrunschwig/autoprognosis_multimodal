"""
Multimodal Studies

Description: This file is a temporary main to test the training of multimodal machine learning model
Author: Lucas Brunschwig (lucas.brunschwig@gmail.com)

"""
# stdlib
from datetime import datetime
import os

# third party
from loader import DataLoader
import psutil

# autoprognosis absolute
import autoprognosis.logger as logger
from autoprognosis.studies.image_classifiers import ImageClassifierStudy

os.environ["N_LEARNER_JOBS"] = "3"
os.environ["N_OPT_JOBS"] = "1"

if __name__ == "__main__":

    logger.debug("Loading Images")

    n = 300

    logger.info(
        f"GB available before loading data: {psutil.virtual_memory().available/1073741824:.2f}"
    )

    DL = DataLoader(
        path_="../data",
        data_src_="PAD-UFES",
        format_="PIL",
    )

    df_train, df_test = DL.load_dataset(sample=True)
    df_train = df_train[["image", "label"]]

    logger.info("Image Loaded")
    logger.info(
        f"GB available after loading data: {psutil.virtual_memory().available/1073741824:.2f}"
    )

    # Study Name
    study_name = f"image_classifier_{datetime.now().strftime('%Y-%m-%H')}"
    study = ImageClassifierStudy(
        study_name=study_name,
        dataset=df_train,  # pandas DataFrame
        target="label",  # the label column in the dataset
        sample_for_search=False,  # no Sampling
        classifiers=["vision_transformer"],
        n_folds_cv=5,
        num_iter=10,
        timeout=36000,
        num_study_iter=10,
    )

    model = study.fit()
