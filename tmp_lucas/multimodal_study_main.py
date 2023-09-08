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
from autoprognosis.studies.multimodal_classifier import MultimodalStudy

os.environ["N_LEARNER_JOBS"] = "1"
os.environ["N_OPT_JOBS"] = "1"

logger.debug("Loading Images")

if __name__ == "__main__":

    logger.info(
        f"GB available before loading data: {psutil.virtual_memory().available/1073741824:.2f}"
    )

    DL = DataLoader(
        path_="../data",
        data_src_="PAD-UFES",
        format_="PIL",
    )
    df_train, df_test = DL.load_dataset(sample=True)

    logger.info("Image Loaded")
    logger.info(
        f"GB available after loading data: {psutil.virtual_memory().available/1073741824:.2f}"
    )

    # Study Name
    multimodal_type = "late_fusion"
    study_name = multimodal_type + f"_{datetime.now().strftime('%Y-%m-%H')}"
    study = MultimodalStudy(
        study_name=study_name,
        dataset=df_train,  # pandas DataFrame
        target="label",  # the label column in the dataset
        image="image",  # the image column in the dataset
        multimodal_type=multimodal_type,
        sample_for_search=False,  # no Sampling
        classifiers=["neural_nets"],
        image_classifiers=["cnn_fine_tune"],
        predefined_cnn=["alexnet"],
        n_folds_cv=5,
        num_iter=1,
        timeout=3600,
        num_study_iter=10,
    )

    model = study.fit()
