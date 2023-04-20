"""
Multimodal Studies

Description: This file is a temporary main to test the training of multimodal machine learning model
Author: Lucas Brunschwig (lucas.brunschwig@gmail.com)

Last Modification: 18.04.2023
"""

from sklearn.datasets import load_breast_cancer
from src.autoprognosis.studies.multimodal import MultimodalStudy  # I have to use relative path instead of the package path
from loader import DataLoader
from autoprognosis.utils.serialization import load_model_from_file
from autoprognosis.utils.tester import evaluate_estimator


DL = DataLoader(path_="/home/enwoute/Documents/master-thesis/data/pad-ufe-20", data_src_="PAD-UFES")


ids, X_images, X_clinic, Y = DL.load_dataset()

df = X_images.join((X_clinic, Y))

study_name = "first test"
study = MultimodalStudy(study_name=study_name,
                        dataset=df,  # pandas DataFrame
                        target="label",  # the label column in the dataset
                        image="images"
                        )

model = study.fit()

# Predict the probabilities of each class using the model
# model.predict_proba(X)
