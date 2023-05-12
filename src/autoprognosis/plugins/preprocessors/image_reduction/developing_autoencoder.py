# # stdlib
# from typing import Any, List, Optional
#
# # third party
# import pandas as pd
# from sklearn.decomposition import PCA
#
# # autoprognosis absolute
# import autoprognosis.plugins.core.params as params
# import autoprognosis.plugins.preprocessors.base as base
# import autoprognosis.utils.serialization as serialization
# from autoprognosis.utils.pip import install
#
#
# for retry in range(2):
#     try:
#         # third party
#         import torch
#         from torch import nn
#         from torch.utils.data import DataLoader, TensorDataset
#
#         break
#     except ImportError:
#         depends = ["torch"]
#         install(depends)
#
#
# class Encoder(nn.Module):
#
#     def __init__(self,
#                  input_dims: tuple,
#                  encoded_space_dim: int,
#                  n_layer, int
#                  ):
#
#         super().__init__()
#
#         n_channels, width, height = input_dims
# #
#         # Reminder: n_out = (n_in + 2p - k)/s
#         #           p = padding, k=kernel size, s = stride
#
#         # (224, 224, 3) -> (112, 112, 4) ->  (64, 64, 4) -> (32, 32, 8) -> (16, 16, 16) -> (8, 8, 32)
#
#         stride_layer_1 = 1
#         kernel_layer_1 = 3
#         padding_layer_1 = int((width/2 * stride_layer_1 - width + kernel_layer_1)/2)
#
#         # Architecture:
#         #   n_in = (3, 224, 224)
#         # Layer 1: convolution(in_channels = 3, out_channels = 8, kernel = 3, stride = 2, padding = 1)
#         #   n_layer = (8, 110, 110)
#
#         # 200 - 300 <-> 3-5 convolutional layer
#
#
#         # Convolutional section
#         self.encoder_cnn = nn.Sequential(
#             nn.Conv2d(n_channels, 8, kernel_layer_1, stride=stride_layer_1, padding=padding_layer_1),
#             nn.ReLU(True),
#             nn.Conv2d(8, 16, 3, stride=2, padding=1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(True),
#             nn.Conv2d(16, 32, 3, stride=2, padding=0),
#             nn.ReLU(True)
#         )
#
#         # Flatten layer
#         self.flatten = nn.Flatten(start_dim=1)
#         # Linear section
#
#         self.encoder_lin = nn.Sequential(
#             nn.Linear(3 * 3 * 32, 128),
#             nn.ReLU(True),
#             nn.Linear(128, encoded_space_dim)
#         )
#
#     def forward(self, x):
#         x = self.encoder_cnn(x)
#         x = self.flatten(x)
#         x = self.encoder_lin(x)
#         return x
#
#
# class Decoder(nn.Module):
#
#     def __init__(self, encoded_space_dim, fc2_input_dim):
#         super().__init__()
#
#         self.decoder_lin = nn.Sequential(
#             nn.Linear(encoded_space_dim, 128),
#             nn.ReLU(True),
#             nn.Linear(128, 3 * 3 * 32),
#             nn.ReLU(True)
#         )
#
#         self.unflatten = nn.Unflatten(dim=1,
#                                       unflattened_size=(32, 3, 3))
#
#         self.decoder_conv = nn.Sequential(
#             nn.ConvTranspose2d(32, 16, 3,
#                                stride=2, output_padding=0),
#             nn.BatchNorm2d(16),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(16, 8, 3, stride=2,
#                                padding=1, output_padding=1),
#             nn.BatchNorm2d(8),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(8, 1, 3, stride=2,
#                                padding=1, output_padding=1)
#         )
#
#     def forward(self, x):
#         x = self.decoder_lin(x)
#         x = self.unflatten(x)
#         x = self.decoder_conv(x)
#         x = torch.sigmoid(x)
#         return x
#
#
# class Autoencoder(nn.Module):
#     """
#     Basic neural net.
#
#     Parameters
#     ----------
#     n_layers_hidden: int
#         Number of hypothesis layers (n_layers_hidden x n_units_hidden + 1 x Linear layer)
#     n_units_hidden: int
#         Number of hidden units in each hypothesis layer
#     nonlin: string, default 'elu'
#         Nonlinearity to use in NN. Can be 'elu', 'relu', 'selu' or 'leaky_relu'.
#     lr: float
#         learning rate for optimizer. step_size equivalent in the JAX version.
#     weight_decay: float
#         l2 (ridge) penalty for the weights.
#     n_iter: int
#         Maximum number of iterations.
#     batch_size: int
#         Batch size
#     n_iter_print: int
#         Number of iterations after which to print updates and check the validation loss.
#     val_split_prop: float
#         Proportion of samples used for validation split (can be 0)
#     patience: int
#         Number of iterations to wait before early stopping after decrease in validation loss
#     n_iter_min: int
#         Minimum number of iterations to go through before starting early stopping
#     clipping_value: int, default 1
#         Gradients clipping value
#     """
#
#     def __init__(
#         self,
#         n_unit_in: int,
#         latent_dim: int,
#         categories_cnt: int,
#         n_layers_hidden: int = 2,
#         n_units_hidden: int = 100,
#         nonlin: str = "relu",
#         lr: float = 1e-3,
#         weight_decay: float = 1e-3,
#         n_iter: int = 1000,
#         batch_size: int = 100,
#         n_iter_print: int = 10,
#         patience: int = 10,
#         n_iter_min: int = 100,
#         dropout: float = 0.1,
#         clipping_value: int = 1,
#         batch_norm: bool = False,
#         early_stopping: bool = True,
#     ) -> None:
#         super(Autoencoder, self).__init__()
#
#         self.encoder = Encoder(latent_dim, 0)
#         self.decoder = Decoder(latent_dim, 0)
#
#         self.n_iter = n_iter
#         self.batch_size = batch_size
#         self.n_iter_print = n_iter_print
#         self.patience = patience
#         self.n_iter_min = n_iter_min
#         self.clipping_value = clipping_value
#         self.early_stopping = early_stopping
#
#         self.optimizer = torch.optim.Adam(
#             params=[{'params': self.encoder.parameters()},{'params': self.decoder.parameters()}],
#             lr=lr,
#             weight_decay=weight_decay
#         )
#
#     def forward(self, X: torch.Tensor) -> torch.Tensor:
#         X = self.encoder(X)
#
#
#         return self.model(X)
#
#     def train(self, X: torch.Tensor, y: torch.Tensor) -> "BasicNet":
#         X = self._check_tensor(X).float()
#         y = self._check_tensor(y).squeeze().long()
#
#         dataset = TensorDataset(X, y)
#
#         train_size = int(0.8 * len(dataset))
#         test_size = len(dataset) - train_size
#         train_dataset, test_dataset = torch.utils.data.random_split(
#             dataset, [train_size, test_size]
#         )
#
#         loader = DataLoader(train_dataset, batch_size=self.batch_size, pin_memory=False)
#
#         # do training
#         val_loss_best = 999999
#         patience = 0
#
#         loss = nn.CrossEntropyLoss()
#
#         for i in range(self.n_iter):
#             train_loss = []
#
#             for batch_ndx, sample in enumerate(loader):
#                 self.optimizer.zero_grad()
#
#                 X_next, y_next = sample
#
#                 preds = self.forward(X_next).squeeze()
#
#                 batch_loss = loss(preds, y_next)
#
#                 batch_loss.backward()
#
#                 torch.nn.utils.clip_grad_norm_(self.parameters(), self.clipping_value)
#
#                 self.optimizer.step()
#
#                 train_loss.append(batch_loss.detach())
#
#             train_loss = torch.Tensor(train_loss).to(DEVICE)
#
#             if self.early_stopping or i % self.n_iter_print == 0:
#                 with torch.no_grad():
#                     X_val, y_val = test_dataset.dataset.tensors
#
#                     preds = self.forward(X_val).squeeze()
#                     val_loss = loss(preds, y_val)
#
#                     if self.early_stopping:
#                         if val_loss_best > val_loss:
#                             val_loss_best = val_loss
#                             patience = 0
#                         else:
#                             patience += 1
#
#                         if patience > self.patience and i > self.n_iter_min:
#                             break
#
#                     if i % self.n_iter_print == 0:
#                         log.trace(
#                             f"Epoch: {i}, loss: {val_loss}, train_loss: {torch.mean(train_loss)}"
#                         )
#
#         return self
#
#     def _check_tensor(self, X: torch.Tensor) -> torch.Tensor:
#         if isinstance(X, torch.Tensor):
#             return X.to(DEVICE)
#         else:
#             return torch.from_numpy(np.asarray(X)).to(DEVICE)
#
#
# class AutoencoderPlugin(base.PreprocessorPlugin):
#     """Preprocessing plugin for dimensionality reduction based on the PCA method.
#
#     Method:
#         PCA is used to decompose a multivariate dataset in a set of successive orthogonal components that explain a maximum amount of the variance.
#
#     Reference:
#         https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
#
#     Args:
#         n_components: int
#             Number of components to use.
#
#     Example:
#         >>> from autoprognosis.plugins.preprocessors import Preprocessors
#         >>> plugin = Preprocessors(category="dimensionality_reduction").get("autoencoder")
#         >>> from sklearn.datasets import load_iris
#         >>> X, y = load_iris(return_X_y=True)
#         >>> plugin.fit_transform(X, y)
#     """
#
#     def __init__(
#         self, random_state: int = 0, model: Any = None, n_components: int = 2
#     ) -> None:
#         super().__init__()
#         self.random_state = random_state
#         self.n_components = n_components
#         self.model: Optional[PCA] = None
#
#         if model:
#             self.model = model
#
#     @staticmethod
#     def name() -> str:
#         return "autoencoder"
#
#     @staticmethod
#     def subtype() -> str:
#         return "dimensionality_reduction"
#
#     @staticmethod
#     def modality_type():
#         return "image"
#
#     @staticmethod
#     def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
#         cmin, cmax = base.PreprocessorPlugin.components_interval(*args, **kwargs)
#         return [params.Integer("n_components", cmin, cmax)]
#
#     def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "PCAPlugin":
#         n_components = min(self.n_components, X.shape[0], X.shape[1])
#
#         self.model = PCA(n_components=n_components, random_state=self.random_state)
#
#         self.model.fit(X, *args, **kwargs)
#
#         return self
#
#     def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
#         return self.model.transform(X)
#
#     def save(self) -> bytes:
#         return serialization.save_model(
#             {"model": self.model, "n_components": self.n_components}
#         )
#
#     @classmethod
#     def load(cls, buff: bytes) -> "PCAPlugin":
#         args = serialization.load_model(buff)
#         return cls(**args)
#
#
# plugin = PCAPlugin
