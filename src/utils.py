import os
import random
from datetime import timedelta
import numpy as np
import json
import torch
import torchvision
from hydra.utils import get_original_cwd
from torchvision.transforms import Compose, ToTensor, Normalize
from src import ffclassifier, ffmodel, classifier
from src import model as md

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


def get_model_and_optimizer(opt):
    if opt.model.name == "ffmodel":
        if opt.model.type == "mlp":
            model = ffmodel.FFModel(opt)
        else:
            model = ffmodel.FFModel(opt)
    else:
        if opt.model.type == "mlp":
            model = md.Model(opt)
        else:
            model = md.Model(opt)

    if "cuda" in opt.device:
        model = model.cuda()

    # Create optimizer with different hyperparameters for the main model
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


def get_data(opt, partition):
    if opt.model.name == "ffmodel":
        if opt.input.dataset == "mnist":
            dataset = ffclassifier.FF_MNIST(opt, partition, num_classes=10)
        elif opt.input.dataset == "cifar10":
            dataset = ffclassifier.FF_CIFAR10(opt, partition, num_classes=10)
    else:
        if opt.input.dataset == "mnist":
            dataset = classifier.MNIST(opt, partition, num_classes=10)
        elif opt.input.dataset == "cifar10":
            dataset = classifier.CIFAR10(opt, partition, num_classes=10)
        else:
            raise ValueError("Unknown dataset.")

    # Improve reproducibility in dataloader.
    g = torch.Generator()
    g.manual_seed(opt.seed)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.input.batch_size,
        drop_last=True,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=1,
        persistent_workers=True,
    )


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_CIFAR10_partition(opt, partition):
    transform = Compose(
        [
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )
    if partition in ["train"]:
        cifar = torchvision.datasets.CIFAR10(
            os.path.join(get_original_cwd(), opt.input.path),
            train=True,
            download=True,
            transform=transform,
        )
    elif partition in ["val", "test"]:
        cifar = torchvision.datasets.CIFAR10(
            os.path.join(get_original_cwd(), opt.input.path),
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
            Normalize((0.1307,), (0.3081,))
        ]
    )
    if partition in ["train"]:
        mnist = torchvision.datasets.MNIST(
            os.path.join(get_original_cwd(), opt.input.path),
            train=True,
            download=True,
            transform=transform,
        )
    elif partition in ["val", "test"]:
        mnist = torchvision.datasets.MNIST(
            os.path.join(get_original_cwd(), opt.input.path),
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
def save_model(model, opt):
    run_name = f'{opt.model.type}-{opt.model.name}_' \
               f'dataset-{opt.input.dataset}'
    torch.save(model.state_dict(), f"{run_name}.pt")

def save_record(record, opt):
    threshold = ''
    if opt.model.name == "ffmodel":
        threshold = f'threshold-{opt.training.threshold}_'

    run_name = f'{opt.model.type}-{opt.model.name}_' \
               f'dataset-{opt.input.dataset}_' \
               f'{threshold}' \
               f'layers-{opt.model.num_layers}_' \
               f'dim-{opt.model.hidden_dim}'
    json_record = json.dumps(record)
    with open(f"archive/{run_name}.json", "w") as f:
        f.write(json_record)

def log_results(result_dict, scalar_outputs, num_steps):
    for key, value in scalar_outputs.items():
        if isinstance(value, float):
            result_dict[key] += value / num_steps
        else:
            result_dict[key] += value.item() / num_steps
    return result_dict
