"""
Multimodal Studies

Description: This file is used to test the inputs and outputs of an early fusion models

Author: Lucas Brunschwig (lucas.brunschwig@gmail.com)

"""

# stdlib
from typing import Optional

# third party
import pandas as pd
import torch
from torch import nn
import torchvision.models as models
import torchvision.transforms as transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pd.set_option("mode.chained_assignment", None)


class EarlyFusion(nn.Module):
    def __init__(
        self,
        n_tabular: int,
        n_layers_hidden: int,
        n_units_hidden: int,
        batch_norm: bool,
        dropout: float,
        categories_cnt: int,
        conv_model: str = None,
    ):
        super(EarlyFusion, self).__init__()

        # Load a predefined CNN model
        if conv_model is not None:
            self.conv_model = ConvNet(
                model_name=conv_model,
                n_features_out=None,
                use_pretrained=True,
                fine_tune=False,
            )

        # Dimension of the classifier
        # TODO: change this to match the size depending on the neural nets
        #       interestingly PyTorch CNN do not assume input size, only number of channels
        self.input_image = self.conv_model(torch.randn((1, 3, 300, 300))).shape[1]
        self.input_tabular = n_tabular
        n_unit_in = self.input_tabular + self.input_image

        if n_layers_hidden > 0:
            if batch_norm:
                layers = [
                    nn.Linear(n_unit_in, n_units_hidden),
                    nn.BatchNorm1d(n_units_hidden),
                    nn.ReLU(),
                ]
            else:
                layers = [nn.Linear(n_unit_in, n_units_hidden), nn.ReLU]

            # add required number of layers
            for i in range(n_layers_hidden - 1):
                if batch_norm:
                    layers.extend(
                        [
                            nn.Dropout(dropout),
                            nn.Linear(n_units_hidden, n_units_hidden),
                            nn.BatchNorm1d(n_units_hidden),
                            nn.ReLU(),
                        ]
                    )
                else:
                    layers.extend(
                        [
                            nn.Dropout(dropout),
                            nn.Linear(n_units_hidden, n_units_hidden),
                            nn.ReLU(),
                        ]
                    )

            # add final layers
            layers.append(nn.Linear(n_units_hidden, categories_cnt))
        else:
            layers = [nn.Linear(n_unit_in, categories_cnt)]

        layers.append(nn.Softmax(dim=-1))

        self.model = nn.Sequential(*layers).to(DEVICE)

    def forward(self, x_img, x_tab):
        # Pass the input image through the Inception model
        x_conv = self.conv_model(x_img)

        # Concatenate the output of the Inception model with the input vector v
        x = torch.cat((x_conv, x_tab), dim=1)

        # Pass the concatenated tensor through the new classifier
        out = self.model(x)

        return out


class ConvNet(nn.Module):
    """Convolutional Neural Network model for early fusion models.

    Parameters
    ----------
    model_name (str):
        model name of one of the default Deep CNN architectures.
    use_pretrained (bool):
        instantiate the model with the latest PyTorch weights of the model.
    fine_tune (bool):
        if false all parameters are set to no grad
    n_features_out (int, optional):
        the number of output features for the ConvNet, if not specified the last layer is replaced by an identity layer.

    TODO:
    - allow user to use their own weight for a given model.

    """

    def __init__(
        self,
        model_name: str,
        use_pretrained: bool,
        n_features_out: Optional[int],
        fine_tune: bool,
    ):

        super(ConvNet, self).__init__()

        self.model_name = model_name
        self.use_pretrained = use_pretrained
        self.n_features_out = n_features_out
        self.fine_tune = fine_tune

        # Lucas: Inception can not be used during fine-tuning due to the auxiliary outputs
        if model_name.lower() == "inception":
            self.model = models.inception_v3(pretrained=use_pretrained)

            # Handle the auxilary net
            num_ftrs = self.model.AuxLogits.fc.in_features
            self.AuxLogits.fc = nn.Linear(num_ftrs, n_features_out)

        elif model_name.lower() == "resnet":
            """Resnet18"""
            self.model = models.resnet18(weights="ResNet18_Weights.DEFAULT")
            self.set_parameter_requires_grad()

        # Replace the output layer by the given number of features
        if n_features_out is not None:
            n_features_in = self.model.fc.in_features
            self.model.fc = nn.Linear(n_features_in, n_features_out)
        else:
            self.model.fc = nn.Identity()

    def set_parameter_requires_grad(
        self,
    ):
        if not self.fine_tune:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":

    # third party
    from loader import DataLoader

    DL = DataLoader(
        path_="/home/enwoute/Documents/master-thesis/data/pad-ufe-20",
        data_src_="PAD-UFES",
        format_="PIL",
    )

    DL.load_dataset()
    X_images, X_clinic, Y = DL.sample_dataset(50)

    # Imputation in the future but currently using 0 for development
    X_clinic.fillna(0, inplace=True)

    # Preprocessing for InceptionV3
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    X_images.images = X_images.images.apply(lambda img_: preprocess(img_))

    model = EarlyFusion(58, 2, 64, True, 0.5, 100, "ResNet")

    prediction = model(
        torch.stack(X_images.images.tolist()), torch.Tensor(X_clinic.to_numpy())
    )
