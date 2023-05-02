"""
Multimodal Studies

Description: This file is a temporary main to test the training of multimodal machine learning model
Author: Lucas Brunschwig (lucas.brunschwig@gmail.com)

"""

# stdlib
import multiprocessing

# third party
from loader import DataLoader

# autoprognosis absolute
import autoprognosis.logger as log
from autoprognosis.studies.multimodal_classifier import MultimodalStudy

log.debug("LOADING IMAGES")


def worker_dataloader(state):
    # Load Data
    DL = DataLoader(
        path_="/home/enwoute/Documents/master-thesis/data/pad-ufe-20",
        data_src_="PAD-UFES",
        format_="PIL",
    )
    state["X_images"], state["X_clinic"], state["Y"] = DL.load_dataset(
        classes=["ACK", "BCC"]
    )

    # Sample Dataset for Testing Purpose
    # state['X_images'], state['X_clinic'], state['Y'] = DL.sample_dataset(state['n'])


if __name__ == "__main__":

    # Multiprocessing for memory issues
    manager = multiprocessing.Manager()
    state = manager.dict(n=1000)
    p = multiprocessing.Process(target=worker_dataloader, args=(state,))
    p.start()
    p.join()
    df = state["X_images"].join((state["X_clinic"], state["Y"]))

    # Study Name
    study_name = "first test"
    study = MultimodalStudy(
        study_name=study_name,
        dataset=df,  # pandas DataFrame
        target="label",  # the label column in the dataset
        image="images",  # the image column in the dataset
    )

    model = study.fit()
