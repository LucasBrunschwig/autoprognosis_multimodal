# third party
import pandas as pd

# autoprognosis absolute
from autoprognosis.utils.pip import install

for retry in range(2):
    try:
        # third party
        import torch
        from torch.utils.data import Dataset

        break
    except ImportError:
        depends = ["torch"]
        install(depends)

for retry in range(2):
    try:
        # third party
        import torchvision

        break
    except ImportError:
        depends = ["torchvision"]
        install(depends)


class TrainingImageDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        target: torch.Tensor,
        preprocess,
        transform: torchvision.transforms.Compose = None,
    ):
        """
        CustomDataset constructor.

        Args:
        images (pd.Dataframe): Dataframe of PIL or Tensor images.
        labels (torch.Tensor): Tensor containing the labels corresponding to the images.
        preprocess (callable): Preprocessing to be applied to image after optional transformation
        transform (callable, optional): Optional transformations to be applied to the images. Default is None.
        """
        self.image = data.squeeze(axis=1)
        self.target = target
        self.transform = transform
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image, label = self.image.iloc[index], self.target[index]

        if self.transform:
            image = self.transform(image)

        image = self.preprocess(image)

        return image, label


class TestImageDataset(Dataset):
    def __init__(self, data, preprocess):
        """
        CustomDataset constructor.

        Args:
        images (pd.DataFrame): Dataframe of PIL or Tensor images.
        labels (torch.Tensor): Tensor containing the labels corresponding to the images.
        preprocess (callable): Preprocessing to be applied to image.
        """
        self.image = pd.DataFrame(data).squeeze(axis=1)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image = self.image.iloc[index]
        image = self.preprocess(image)

        return image
