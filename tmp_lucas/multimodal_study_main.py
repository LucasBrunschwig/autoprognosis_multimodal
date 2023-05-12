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

    subprocess = False
    n = 100

    # Use a subprocess to free memory
    if subprocess:

        manager = multiprocessing.Manager()

        state = manager.dict(n=n)
        p = multiprocessing.Process(target=worker_dataloader, args=(state,))
        p.start()
        p.join()
        df = state["X_images"].join((state["X_clinic"], state["Y"]))

    else:
        DL = DataLoader(
            path_="/home/enwoute/Documents/master-thesis/data/pad-ufe-20",
            data_src_="PAD-UFES",
            format_="PIL",
        )
        DL.load_dataset(classes=["ACK", "BCC"])

        # Sample Dataset for Testing Purpose
        X_images, X_clinic, Y = DL.sample_dataset(n)

        df = X_images.join((X_clinic, Y))

    # Study Name
    study_name = "first test"
    study = MultimodalStudy(
        study_name=study_name,
        dataset=df,  # pandas DataFrame
        target="label",  # the label column in the dataset
        image="image",  # the image column in the dataset
        sample_for_search=False,  # no Sampling
    )

    model = study.fit()
