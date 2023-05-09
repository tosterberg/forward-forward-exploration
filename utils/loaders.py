import torch
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader


def mnist_loaders(train_size, test_size, norms):
    transform = Compose([ToTensor(), Normalize(*norms), Lambda(lambda x: torch.flatten(x))])
    train_loader = DataLoader(MNIST('../data/',
                                    train=True,
                                    download=True,
                                    transform=transform),
                              batch_size=train_size,
                              shuffle=True)

    test_loader = DataLoader(MNIST('../data/',
                                   train=True,
                                   download=True,
                                   transform=transform),
                             batch_size=test_size,
                             shuffle=True)

    return train_loader, test_loader
