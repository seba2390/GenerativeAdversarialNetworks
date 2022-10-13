import torch
from math import prod
from Util import *


class Generator(torch.nn.Module):
    """ check https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout
     --> use normalization directly after layer (before activation function) and dropout after activation."""

    def __init__(self, latent_dims: int = 100, hidden_dims: int = 128, image_dims: tuple[int, int] = (28, 28)):
        super(Generator, self).__init__()
        # Random value vector size
        self.latent_dims = latent_dims
        self.hidden_dims = hidden_dims
        self.image_dims = image_dims
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # latent_dims => 28*28 (product of image_dims)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.latent_dims, out_features=prod(self.image_dims)),
            torch.nn.BatchNorm1d(num_features=prod(self.image_dims)),
            torch.nn.LeakyReLU(0.01),
        )

        # 28*28 (product of image_dims) => 16 x 7 x 7
        self.reshape = Reshape(16, 7, 7)

        # 16 X 7 X 7 => 32 x 14 x 14
        self.conv1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=16, out_channels=32, padding=(2, 2),
                                     kernel_size=(5, 5), stride=(2, 2),
                                     output_padding=(1, 1), bias=False),
            torch.nn.BatchNorm2d(num_features=32),
            torch.nn.LeakyReLU(negative_slope=0.01)
        )

        # 32 x 14 x 14 => 1 x 28 x 28 (image_dims)
        self.conv2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=32, out_channels=1, padding=(2, 2),
                                     kernel_size=(5, 5), stride=(2, 2),
                                     output_padding=(1, 1), bias=False),
            torch.nn.Sigmoid()
        )
        self.to(self.device)

    def forward(self, latent_vectors: torch.Tensor):
        assert len(latent_vectors.shape) == 2, "Batch of latens vector should have shape: (batch size, latent_dims)"
        assert latent_vectors.shape[
                   1] == self.latent_dims, f'Each latent vector in batch should be of size: {self.latent_dims}'
        # Forward pass
        generated_images = self.fc(latent_vectors)
        generated_images = self.reshape(generated_images)
        generated_images = self.conv1(generated_images)
        generated_images = self.conv2(generated_images)
        return generated_images
