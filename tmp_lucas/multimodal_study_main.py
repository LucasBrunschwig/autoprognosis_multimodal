"""
Multimodal Studies

Description: This file is a temporary main to test the training of multimodal machine learning model
Author: Lucas Brunschwig (lucas.brunschwig@gmail.com)

Last Modification: 18.04.2023
"""

# third party
from loader import DataLoader

# autoprognosis absolute
from autoprognosis.studies.multimodal_classifier import MultimodalStudy

DL = DataLoader(
    path_="/home/enwoute/Documents/master-thesis/data/pad-ufe-20", data_src_="PAD-UFES"
)


ids, X_images, X_clinic, Y = DL.load_dataset()

df = X_images.join((X_clinic, Y))

study_name = "first test"
study = MultimodalStudy(
    study_name=study_name,
    dataset=df,  # pandas DataFrame
    target="label",  # the label column in the dataset
    image="images",
)

model = study.fit()

# Predict the probabilities of each class using the model
# model.predict_proba(X)
