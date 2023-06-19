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
from autoprognosis.studies.classifiers import ClassifierStudy

os.environ["N_LEARNER_JOBS"] = "3"
os.environ["N_OPT_JOBS"] = "1"

if __name__ == "__main__":

    logger.debug("Loading Images")

    n = 1000

    logger.info(
        f"GB available before loading data: {psutil.virtual_memory().available/1073741824:.2f}"
    )

    # Use a subprocess to free memory

    DL = DataLoader(
        path_="/home/enwoute/Documents/master-thesis/data/pad-ufe-20",
        data_src_="PAD-UFES",
        format_="PIL",
    )

    DL.load_dataset()
    df = DL.sample_dataset(n)
    # Sample Dataset for Testing Purpose
    df = df.drop(["image"], axis=1)
    logger.info("Image Loaded")
    logger.info(
        f"GB available after loading data: {psutil.virtual_memory().available/1073741824:.2f}"
    )

    # Study Name
    study_name = f"tabular_classifier_{datetime.now().strftime('%Y-%m-%H')}"
    study = ClassifierStudy(
        study_name=study_name,
        dataset=df,  # pandas DataFrame
        target="label",  # the label column in the dataset
        sample_for_search=False,  # no Sampling
        n_folds_cv=5,
        num_iter=2,
        classifiers=["neural_nets", "linear_svm"],
        timeout=3600,
        num_study_iter=2,
    )

    model = study.fit()
