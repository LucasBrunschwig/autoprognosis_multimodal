"""
Multimodal Studies

Description: This file is a temporary main to test the grad-cam explainers
Author: Lucas Brunschwig (lucas.brunschwig@gmail.com)

"""


# third party
from loader import DataLoader
import psutil

# autoprognosis absolute
import autoprognosis.logger as logger
from autoprognosis.plugins.explainers import Explainers
from autoprognosis.studies.image_classifiers import ImageClassifierStudy
from autoprognosis.utils.serialization import load_model_from_file

if __name__ == "__main__":

    train_model = True
    explain = True

    logger.debug("Loading Images")

    n = 300

    logger.info(
        f"GB available before loading data: {psutil.virtual_memory().available/1073741824:.2f}"
    )

    # Use a subprocess to free memory

    DL = DataLoader(
        path_="../data",
        data_src_="PAD-UFES",
        format_="PIL",
    )

    df = DL.load_dataset()
    df = df[["image", "label"]]

    logger.info("Image Loaded")
    logger.info(
        f"GB available after loading data: {psutil.virtual_memory().available/1073741824:.2f}"
    )

    if train_model:
        # Study Name
        study_name = "test_gradcam_alexnet"
        study = ImageClassifierStudy(
            study_name=study_name,
            dataset=df,  # pandas DataFrame
            target="label",  # the label column in the dataset
            sample_for_search=False,  # no Sampling
            n_folds_cv=5,
            num_iter=1,
            timeout=36000,
            num_study_iter=1,
            classifiers=["cnn"],
            workspace=r"C:\Users\Lucas\Desktop\Master-Thesis-Cambridge\autoprognosis_multimodal\tmp_lucas",
        )

        model = study.fit()

        print(model)

    if explain:
        best_model = load_model_from_file(
            r"C:\Users\Lucas\Desktop\Master-Thesis-Cambridge\autoprognosis_multimodal\tmp_lucas\test_gradcam_alexnet\model_trained.p"
        )

        explainers = Explainers().get(
            "grad_cam",
            estimator=best_model,
            target_layer="avgpool",
            X=df[["image"]],
            y=df[["label"]],
            task_type="classification",
            prefit=True,
        )
        explainers.explain(
            df[["image"]],
            df[["label"]],
        )
