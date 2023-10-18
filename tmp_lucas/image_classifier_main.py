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
from autoprognosis.studies.image_classifiers import ImageClassifierStudy

logger.add(sink=sys.stdout, level="INFO")

os.environ["N_LEARNER_JOBS"] = "1"
os.environ["N_OPT_JOBS"] = "1"

if __name__ == "__main__":

    sample = False
    group_stratification = True  # use without sampling

    logger.debug("Loading Images")

    logger.info(
        f"GB available before loading data: {psutil.virtual_memory().available/1073741824:.2f}"
    )

    DL = DataLoader(
        path_="../data",
        data_src_="PAD-UFES",
        format_="PIL",
    )
    df_train, df_test = DL.load_dataset(sample=sample)

    logger.info(
        f"GB available after loading data: {psutil.virtual_memory().available/1073741824:.2f}"
    )

    if group_stratification:
        group = ["_".join(patient.split("_")[0:2]) for patient in list(df_train.index)]
        df_train["patient"] = group

    df_train.reset_index(drop=True, inplace=True)
    df_train = df_train[["image", "label"]]
    df_test.reset_index(drop=True, inplace=True)
    df_test = df_test[["image", "label"]]

    # Study Name
    study_name = f"tabular_classifier_{datetime.now().strftime('%Y-%m-%H')}"

    # Study Name
    study_name = f"image_classifier_{datetime.now().strftime('%Y-%m-%H')}"
    study = ImageClassifierStudy(
        study_name=study_name,
        predefined_cnn=["resnet34"],
        dataset=df_train,  # pandas DataFrame
        target="label",  # the label column in the dataset
        sample_for_search=False,  # no Sampling
        classifiers=["cnn", "cnn_fine_tune"],
        n_folds_cv=5,
        num_iter=1,
        timeout=36000,
        num_study_iter=1,
    )

    model = study.fit()
