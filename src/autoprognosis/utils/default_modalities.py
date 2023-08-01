# stdlib
from typing import Optional

# third party
import pandas as pd

IMAGE_KEY = "img"
TABULAR_KEY = "tab"
MULTIMODAL_KEY = "multimodal"


def dataset_to_multimodal(
    dataset: pd.DataFrame,
    label: str,
    image: list,
    tabular: Optional[list] = [],
):

    dataset = dataset.copy(deep=True)

    if label in dataset.columns:
        df_label = dataset[[label]].squeeze()
        dataset.drop([label], axis=1, inplace=True)
    else:
        raise ValueError("Target not in dataset")

    X = {IMAGE_KEY: [], TABULAR_KEY: []}
    if not isinstance(image, list):
        image = [image]
    X[IMAGE_KEY] = dataset[image]

    if len(tabular) == 0:
        tabular = dataset.columns.difference(image)
    X[TABULAR_KEY] = dataset[tabular]

    return X, df_label
