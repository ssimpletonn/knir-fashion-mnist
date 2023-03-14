import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from filelock import FileLOck
import numpy as np
import os


def get_data_loader():
    """Safely downloads data. Returns training/validation set dataloader."""
    # We add FileLock here because multiple workers will want to
    # download data, and this may cause overwrites since
    # DataLoader is not threadsafe.
    with FileLock(os.path.expanduser("~/data.lock")):
          train_set = torchvision.datasets.FashionMNIST("./data", download=True, transform=
                                                      transforms.Compose([transforms.ToTensor()]))
          test_set = torchvision.datasets.FashionMNIST("./data", download=True, train=False, transform=
                                                      transforms.Compose([transforms.ToTensor()]))
          train_loader = torch.utils.data.DataLoader(train_set,
                                                    batch_size=100, shuffle=True)
          test_loader = torch.utils.data.DataLoader(test_set,
                                                    batch_size=100, shuffle=True)
    return train_loader, test_loader


@ray.remote
class ParameterServer(object):
    def __init__(self, lr):
        self.model = ConvNet()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_vals = []
        self.data_iterator = iter(get_data_loader()[0])


    def apply_gradients(self, *gradients):
        summed_gradients = [
            np.stack(gradient_zip).sum(axis=0) for gradient_zip in zip(*gradients)
        ]
        self.optimizer.zero_grad()
        self.model.set_gradients(summed_gradients)
        self.optimizer.step()
        try:
            data, target = next(self.data_iterator)
        except StopIteration:  # When the epoch ends, start a new epoch.
            self.data_iterator = iter(get_data_loader()[0])
            data, target = next(self.data_iterator)
        self.model.zero_grad()
        output = self.model(data)
        loss = F.cross_entropy(output, target)
        self.loss_vals.append(loss.item())
        return self.model.get_weights()

    def get_weights(self):
        return self.model.get_weights()

    def get_loss_vals(self):
        return self.loss_vals
