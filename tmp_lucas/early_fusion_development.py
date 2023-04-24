"""
Multimodal Studies

Description: This file is used to test the inputs and outputs of an early fusion models

Author: Lucas Brunschwig (lucas.brunschwig@gmail.com)

"""

# third party
import torch
from torch import nn
import torchvision.models as models
import torchvision.transforms as transforms


class EarlyFusion(nn.Module):
    def __init__(
        self,
        n_dim: int,
        n_hidden_layer: int,
        n_hidden_units: int,
        n_classes: int,
        conv_model: str = None,
    ):
        super(EarlyFusion, self).__init__()

        # Load a predefined CNN model
        if conv_model is not None:
            self.conv_model = ConvNet(
                model_name=conv_model,
                n_features_out=1000,
                use_pretrained=True,
                feature_extract=True,
                fine_tune=False,
            )

        # Replace the last layer with an identity layer
        self.conv_model.fc = nn.Identity()

        # Define the additional fully connected layer to concatenate with the output of the Inception model
        self.classifier = nn.Linear(n_dim + self.conv_model.fc.out_features, n_classes)

    def forward(self, x, v):
        # Pass the input image through the Inception model
        x = self.conv_model(x)

        # Concatenate the output of the Inception model with the input vector v
        x = torch.cat((x, v), dim=1)

        # Pass the concatenated tensor through the additional fully connected layer
        x = self.fc(x)

        return x


class ConvNet(nn.Module):
    def __init__(
        self,
        model_name: str,
        n_features_out: int,
        use_pretrained: bool,
        feature_extract: bool,
        fine_tune: bool,
    ):

        super(ConvNet, self).__init__()

        # Lucas: Inception can not be used during fine-tuning due to the auxiliary outputs
        if model_name.lower() == "inception":
            model_ft = models.inception_v3(pretrained=use_pretrained)

            # Handle the auxilary net
            num_ftrs = model_ft.AuxLogits.fc.in_features
            model_ft.AuxLogits.fc = nn.Linear(num_ftrs, n_features_out)

            # Handle the primary net
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, n_features_out)

        elif model_name.lower() == "resnet":
            """Resnet18"""
            self.model = models.resnet18(pretrained=use_pretrained)
            self.set_parameter_requires_grad(self.model, feature_extract)

            # Not sure, we want to fine tune this
            if fine_tune:
                n_features_in = self.model.fc.in_features
                self.model.fc = nn.Linear(n_features_in, n_features_out)

    @classmethod
    def set_parameter_requires_grad(cls, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    # third party
    from loader import DataLoader

    DL = DataLoader(
        path_="/home/enwoute/Documents/master-thesis/data/pad-ufe-20",
        data_src_="PAD-UFES",
        format_="Tensor",
    )

    DL.load_dataset()
    X_images, X_clinic, Y = DL.sample_dataset()

    # Preprocessing for InceptionV3
    preprocess = transforms.Compose(
        [
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    X_images.images = X_images.images.apply(
        lambda img_: preprocess(img_.to(torch.float32))
    )

    model = EarlyFusion(58)

    prediction = model(
        torch.stack(X_images.images.tolist()), torch.Tensor(X_clinic.to_numpy())
    )
