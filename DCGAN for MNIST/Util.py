import os
import numpy as np
import torch


# Reshape helper
class Reshape(torch.nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(-1, *self.shape)


def save_config_dict(config_dict: dict, filename: str = "my_config.txt"):
    path = os.getcwd() + "/wandb/latest-run/files/" + filename
    print("saved at;:", path)
    with open(path, 'w') as file:
        file.write("#### RUN CONFIGURATIONS ####\n")
        for key in list(config_dict.keys()):
            file.write(str(key) + ": " + str(config_dict[key]) + "\n")
        file.close()


def keep_numbers(numbers: list[int, ...], dataset) -> torch.Tensor:
    a = np.array([(dataset.targets == i).numpy() for i in numbers])
    keeps = [False for i in range(len(a[0]))]
    for data_point in range(len(a[0])):
        for number in range(len(numbers)):
            if a[:, data_point][number]:
                keeps[data_point] = True
    return torch.tensor(keeps, dtype=None)

