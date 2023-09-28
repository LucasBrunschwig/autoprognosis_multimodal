# stdlib
from typing import Union

# third party
import pandas as pd

# autoprognosis absolute
import autoprognosis.logger as log
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
        from torchvision import transforms

        break
    except ImportError:
        depends = ["torchvision"]
        install(depends)

DATA_AUGMENTATION = {
    "": None,
    "autoaugment_imagenet": [
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET)
    ],
    "autoaugment_cifar10": [
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10)
    ],
    "rand_augment": [transforms.RandAugment()],
    "trivial_augment": [transforms.TrivialAugmentWide()],
    "simple_strategy": [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomResizedCrop(224),
        transforms.RandomRotation(10),
    ],
    "gaussian_noise": [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomResizedCrop(224),
        transforms.RandomRotation(10),
        transforms.GaussianBlur(3, sigma=(0.1, 0.5)),
    ],
    "color_jittering": [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomResizedCrop(224),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05),
    ],
}

data_augmentation_strategies = list(DATA_AUGMENTATION.keys())


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


def build_data_augmentation_strategy(
    data_augmentation_: Union[str, transforms.Compose, None] = None
) -> Union[torchvision.transforms.Compose, None]:
    """This function returns the selected data augmentation strategy. The data augmentation can be either a custom
    composition of transforms, one of the predefined strategy in the data augmentation dictionary, or none.
    """
    if isinstance(data_augmentation_, str):
        if data_augmentation_ == "":
            return None
        elif data_augmentation_ in DATA_AUGMENTATION.keys():
            return transforms.Compose(DATA_AUGMENTATION[data_augmentation_])
        else:
            raise ValueError(
                f"Unknown Data Augmentation Strategy: {data_augmentation_}"
            )
    elif isinstance(data_augmentation_, transforms.Compose):
        return data_augmentation_
    else:
        log.warning("Data Augmentation set to None")
        return None
