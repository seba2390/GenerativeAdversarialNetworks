import torch
from math import prod


# Discriminator Network
class Discriminator(torch.nn.Module):
    def __init__(self, image_dims: tuple[int, int] = (28, 28), hidden_dims: int = 128):
        super(Discriminator, self).__init__()

        # MNIST digit pictures are 28 x 28.
        self.image_dims = image_dims
        self.hidden_dims = hidden_dims
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 1 x 28 x 28 => 32 x 14 x 14
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5),
                            stride=(2, 2), padding=(2, 2), bias=False),
            torch.nn.LeakyReLU(negative_slope=0.01)
        )

        # 32 x 14 x 14 => 16 x 7 x 7
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32,out_channels=16,kernel_size=(5,5),
                            stride=(2,2),padding=(2,2),bias=False),
            torch.nn.LeakyReLU(negative_slope=0.01)
        )

        # 16 x 7 x 7 => 28*28 (product of image_dims)
        self.fc = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=prod(self.image_dims),
                            out_features=prod(self.image_dims)),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Linear(in_features=prod(self.image_dims),
                            out_features=1)
        )
        self.to(self.device)

    def forward(self, images: torch.Tensor):
        """Assuming batched data"""
        assert len(
            images.shape) == 4, f'Images should be given as shape: (batch_size, nr_channels, height, width), but is {images.shape}'
        assert images.shape[
               2:] == self.image_dims, f'Dimension of each image in batch should be: {self.image_dims}, but is {images.shape[2:]}'
        prediction = self.conv1(images)
        prediction = self.conv2(prediction)
        prediction = self.fc(prediction)
        return prediction

    @staticmethod
    def loss(targets: torch.Tensor, predictions: torch.Tensor):
        return torch.nn.functional.binary_cross_entropy_with_logits(input=predictions,
                                                                    target=targets)
