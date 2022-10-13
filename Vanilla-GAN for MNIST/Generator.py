import torch
from math import prod


class Generator(torch.nn.Module):
    """ check https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout
     --> use normalization directly after layer (before activation function) and dropout after activation."""
    def __init__(self, latent_dims: int = 100, hidden_dims: int = 128, image_dims: tuple[int, int] = (28, 28)):
        super(Generator, self).__init__()
        # Random value vector size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.latent_dims = latent_dims
        self.hidden_dims = hidden_dims
        self.image_dims  = image_dims
        self.network = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.latent_dims, out_features=self.hidden_dims),
            # torch.nn.BatchNorm1d(num_features=self.hidden_dims),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(in_features=self.hidden_dims, out_features=prod(self.image_dims)),
            # torch.nn.BatchNorm1d(num_features=prod(self.image_dims)),
            torch.nn.Sigmoid(),
            torch.nn.Dropout(0.4)
        )
        self.to(self.device)

    def forward(self, latent_vectors: torch.Tensor):
        assert len(latent_vectors.shape) == 2, "Batch of latens vector should have shape: (batch size, latent_dims)"
        assert latent_vectors.shape[
                   1] == self.latent_dims, f'Each latent vector in batch should be of size: {self.latent_dims}'
        # Forward pass
        generated_images = self.network(latent_vectors)
        # Reshaping flattened output to array of image dims
        generated_images = generated_images.reshape(latent_vectors.shape[0], 1, self.image_dims[0], self.image_dims[1])
        return generated_images

