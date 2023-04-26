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

# Load Data
DL = DataLoader(
    path_="/home/enwoute/Documents/master-thesis/data/pad-ufe-20",
    data_src_="PAD-UFES",
    format_="PIL",
)
DL.load_dataset()

# Sample Dataset for Testing Purpose
X_images, X_clinic, Y = DL.sample_dataset(n=50)
df = X_images.join((X_clinic, Y))

# Study Name
study_name = "first test"
study = MultimodalStudy(
    study_name=study_name,
    dataset=df,  # pandas DataFrame
    target="label",  # the label column in the dataset
    image="images",  # tje images column in the dataset
)

model = study.fit()

# Predict the probabilities of each class using the model
# model.predict_proba(X)
