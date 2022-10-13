import torch
from math import prod


# Discriminator Network
class Discriminator(torch.nn.Module):
    def __init__(self, image_dims: tuple[int, int] = (28, 28), hidden_dims: int = 128):
        super(Discriminator, self).__init__()

        # MNIST digit pictures are 28 x 28.
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.image_dims  = image_dims
        self.hidden_dims = hidden_dims
        self.network = torch.nn.Sequential(
            torch.nn.Linear(in_features=prod(self.image_dims), out_features=self.hidden_dims),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(in_features=self.hidden_dims, out_features=1),
            torch.nn.Dropout(0.4)
        )
        self.to(self.device)

    def forward(self, images: torch.Tensor):
        """Assuming batched data"""
        assert len(images.shape) == 4, f'Images should be given as shape: (batch_size, nr_channels, height, width), but is {images.shape}'
        assert images.shape[
               2:] == self.image_dims, f'Dimension of each image in batch should be: {self.image_dims}, but is {images.shape[2:]}'
        images = images.reshape(-1, prod(self.image_dims))  # Flattening each image in batch
        prediction = self.network(images)
        return prediction

    @staticmethod
    def loss(targets: torch.Tensor, predictions: torch.Tensor):
        return torch.nn.functional.binary_cross_entropy_with_logits(input=predictions,
                                                                    target=targets)
