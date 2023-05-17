import os
import torch
import random
import torchvision
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
from datetime import timedelta
import numpy as np
import pandas as pd

DATA = '../data/'


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


def parse_args(opt):
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)
    return opt


def get_input_layer_size(opt):
    if opt.input.dataset == "mnist":
        return 784
    elif opt.input.dataset == "cifar10":
        return 3072
    else:
        raise ValueError("Unknown dataset.")


def get_model_and_optimizer(opt, model):
    if "cuda" in opt.device:
        model = model.cuda()
    print(model, "\n")

    # Create optimizer with different hyper-parameters for the main model
    # and the downstream classification model.
    main_model_params = [
        p
        for p in model.parameters()
        if all(p is not x for x in model.classification_loss.parameters())
    ]
    optimizer = torch.optim.SGD(
        [
            {
                "params": main_model_params,
                "lr": opt.training.learning_rate,
                "weight_decay": opt.training.weight_decay,
                "momentum": opt.training.momentum,
            },
            {
                "params": model.classification_loss.parameters(),
                "lr": opt.training.downstream_learning_rate,
                "weight_decay": opt.training.downstream_weight_decay,
                "momentum": opt.training.momentum,
            },
        ]
    )
    return model, optimizer
# 784, 2000, 2000, 2000 # main params
# 6000, 10 # classification_loss params


def get_CIFAR10_partition(opt, partition):
    transform = Compose(
        [
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )
    if partition in ["train"]:
        cifar = torchvision.datasets.CIFAR10(
            DATA,
            train=True,
            download=True,
            transform=transform,
        )
    elif partition in ["val", "test"]:
        cifar = torchvision.datasets.CIFAR10(
            DATA,
            train=False,
            download=True,
            transform=transform,
        )
    else:
        raise NotImplementedError

    return cifar


def get_MNIST_partition(opt, partition):
    transform = Compose(
        [
            ToTensor(),
        ]
    )
    if partition in ["train"]:
        mnist = torchvision.datasets.MNIST(
            DATA,
            train=True,
            download=True,
            transform=transform,
        )
    elif partition in ["val", "test"]:
        mnist = torchvision.datasets.MNIST(
            DATA,
            train=False,
            download=True,
            transform=transform,
        )
    else:
        raise NotImplementedError

    return mnist


def dict_to_cuda(dict):
    for key, value in dict.items():
        dict[key] = value.cuda(non_blocking=True)
    return dict


def preprocess_inputs(opt, inputs, labels):
    if "cuda" in opt.device:
        inputs = dict_to_cuda(inputs)
        labels = dict_to_cuda(labels)
    return inputs, labels


# cools down after the first half of the epochs
def get_linear_cooldown_lr(opt, epoch, lr):
    if epoch > (opt.training.epochs // 2):
        return lr * 2 * (1 + opt.training.epochs - epoch) / opt.training.epochs
    else:
        return lr


def update_learning_rate(optimizer, opt, epoch):
    optimizer.param_groups[0]["lr"] = get_linear_cooldown_lr(
        opt, epoch, opt.training.learning_rate
    )
    optimizer.param_groups[1]["lr"] = get_linear_cooldown_lr(
        opt, epoch, opt.training.downstream_learning_rate
    )
    return optimizer


def get_accuracy(opt, output, target):
    """Computes the accuracy."""
    with torch.no_grad():
        prediction = torch.argmax(output, dim=1)
        return (prediction == target).sum() / opt.input.batch_size


def print_results(partition, iteration_time, scalar_outputs, epoch=None):
    if epoch is not None:
        print(f"Epoch {epoch} \t", end="")

    print(
        f"{partition} \t \t"
        f"Time: {timedelta(seconds=iteration_time)} \t",
        end="",
    )
    if scalar_outputs is not None:
        for key, value in scalar_outputs.items():
            print(f"{key}: {value:.4f} \t", end="")
    print()
    partition_scalar_outputs = {}
    if scalar_outputs is not None:
        for key, value in scalar_outputs.items():
            partition_scalar_outputs[f"{partition}_{key}"] = value


# create save_model function
def save_model(model):
    torch.save(model.state_dict(), f"{random.choice([range(100)])}-model.pt")


def log_results(result_dict, scalar_outputs, num_steps):
    for key, value in scalar_outputs.items():
        if isinstance(value, float):
            result_dict[key] += value / num_steps
        else:
            result_dict[key] += value.item() / num_steps
    return result_dict