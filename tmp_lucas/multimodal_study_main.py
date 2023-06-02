"""
Multimodal Studies

Description: This file is a temporary main to test the training of multimodal machine learning model
Author: Lucas Brunschwig (lucas.brunschwig@gmail.com)

"""
# stdlib
import multiprocessing
import os

# third party
from loader import DataLoader
import psutil

# autoprognosis absolute
import autoprognosis.logger as logger
from autoprognosis.studies.multimodal_classifier import MultimodalStudy

os.environ["N_LEARNER_JOBS"] = "3"
os.environ["N_OPT_JOBS"] = "1"

logger.debug("Loading Images")


def worker_dataloader(state):
    # Load Data
    DL = DataLoader(
        path_="/home/enwoute/Documents/master-thesis/data/pad-ufe-20",
        data_src_="PAD-UFES",
        format_="PIL",
    )
    DL.load_dataset(classes=["ACK", "BCC"])

    # Sample Dataset for Testing Purpose
    state["X_images"], state["X_clinic"], state["Y"] = DL.sample_dataset(state["n"])


if __name__ == "__main__":

    subprocess = False
    n = 1000

    logger.debug(
        f"GB available before loading data: {psutil.virtual_memory().available/1073741824:.2f}"
    )

    # Use a subprocess to free memory
    if subprocess:

        manager = multiprocessing.Manager()

        state = manager.dict(n=n)
        p = multiprocessing.Process(target=worker_dataloader, args=(state,))
        p.start()
        p.join()
        df = state["df"]

    else:
        DL = DataLoader(
            path_="/home/enwoute/Documents/master-thesis/data/pad-ufe-20",
            data_src_="PAD-UFES",
            format_="PIL",
        )
        DL.load_dataset()
        df = DL.sample_dataset(n)
        # Sample Dataset for Testing Purpose

    logger.debug("Image Loaded")
    logger.debug(
        f"GB available after loading data: {psutil.virtual_memory().available/1073741824:.2f}"
    )

    # Study Name
    study_name = "first test"
    study = MultimodalStudy(
        study_name=study_name,
        dataset=df,  # pandas DataFrame
        target="label",  # the label column in the dataset
        image="image",  # the image column in the dataset
        multimodal_type="late_fusion",
        sample_for_search=False,  # no Sampling
        image_dimensionality_reduction=["predefined_cnn"],
        n_folds_cv=5,
        num_iter=10,
        timeout=360,
        num_study_iter=50,
    )

    model = study.fit()
