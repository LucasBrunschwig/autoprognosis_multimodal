"""
Multimodal Studies

Description: This file is a temporary main to test the training of multimodal machine learning model
Author: Lucas Brunschwig (lucas.brunschwig@gmail.com)

"""
# stdlib
from datetime import datetime
import os
import sys

# third party
from loader import DataLoader
import psutil

# autoprognosis absolute
import autoprognosis.logger as logger
from autoprognosis.studies.multimodal_classifier import MultimodalStudy

os.environ["N_LEARNER_JOBS"] = "1"
os.environ["N_OPT_JOBS"] = "1"

logger.add(sink=sys.stdout, level="INFO")

logger.debug("Loading Images")


if __name__ == "__main__":

    sample = True
    group_stratification = False  # use without sampling

    logger.debug("Loading Images")

    logger.info(
        f"GB available before loading data: {psutil.virtual_memory().available / 1073741824:.2f}"
    )

    DL = DataLoader(
        path_="../data",
        data_src_="PAD-UFES",
        format_="PIL",
    )
    df_train, df_test = DL.load_dataset(sample=sample)

    if group_stratification:
        group = ["_".join(patient.split("_")[0:2]) for patient in list(df_train.index)]
        df_train["patient"] = group

    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    logger.info("Image Loaded")
    logger.info(
        f"GB available after loading data: {psutil.virtual_memory().available/1073741824:.2f}"
    )

    # Study Name
    multimodal_type = "intermediate_fusion"
    # LATE FUSION: tabular_classifiers, image_classifiers
    # EARLY FUSION: tabular_classifiers, image_dimensionality_reduction, fusion
    # INTERMEDIATE FUSION: intermediate_classifiers

    study_name = multimodal_type + f"_{datetime.now().strftime('%Y-%m-%H')}"
    study = MultimodalStudy(
        study_name=study_name,
        dataset=df_train,  # pandas DataFrame
        target="label",  # the label column in the dataset
        image="image",  # the image column in the dataset
        multimodal_type=multimodal_type,
        sample_for_search=False,
        # Different Models for different type of fusion
        intermediate_classifiers=["metablock", "intermediate_conv_net"],
        tabular_classifiers=["logistic_regression", "random_forest"],
        image_classifiers=["cnn_fine_tune", "cnn"],
        # For early fusion
        image_dimensionality_reduction=["cnn_fine_tune", "cnn_imagenet"],
        fusion=["concatenate"],
        # Miscellaneous
        predefined_cnn=["alexnet"],
        n_folds_cv=3,
        num_iter=1,
        timeout=3600,
        num_study_iter=1,
    )

    model = study.fit()
