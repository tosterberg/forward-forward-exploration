import torch
import os
import argparse
import json
from dotwiz import DotWiz
from networks import FFDenseNet
from functions import embed_label
from utils import mnist_loaders
from layers import FFLayer
import matplotlib.pyplot as plt

MNIST_NORMS = ((0.1307,), (0.3081,))
IN_DIM = 784
HIDDEN_DIM = 500
LAYERS = 2


def load_config(filename):
    with open(os.path.join(os.getcwd(), filename), "r") as file:
        return json.load(file)


def format_config(dictionary):
    return DotWiz(dictionary)


def visualize_sample(data, name='', idx=0):
    reshaped = data[idx].cpu().reshape(28, 28)
    plt.figure(figsize=(4, 4))
    plt.title(name)
    plt.imshow(reshaped, cmap="gray")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default=None, help="Load model config from a config file"
    )
    parser.add_argument(
        "--no_cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--no_mps", action="store_true", default=False, help="disables MPS training"
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--train_size", type=int, default=50_000, help="size of training set"
    )
    parser.add_argument("--test_size", type=int, default=10_000, help="size of test set")
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {"batch_size": args.train_size}
    test_kwargs = {"batch_size": args.test_size}

    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    torch.manual_seed(args.seed)
    config_dict = dict()
    config_dict["model"] = {
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "threshold": args.threshold,
        "train_size": args.train_size,
        "test_size": args.test_size,
        "seed": args.seed
    }
    if args.config:
        config_dict["model"] = load_config(args.config)
    config = format_config(config_dict)

    layer_config = {}

    train_loader, test_loader = mnist_loaders(args.train_size, args.test_size, MNIST_NORMS)

    model = FFDenseNet([IN_DIM] + [HIDDEN_DIM] * LAYERS, layer=FFLayer, layer_config=layer_config, device=device)
    x_train, y_train = next(iter(train_loader))
    x_train, y_train = x_train.to(device), y_train.to(device)
    positive_x = embed_label(x_train, y_train)
    random_label = torch.randperm(x_train.size(0))
    negative_x = embed_label(x_train, y_train[random_label])

    # for data, name in zip([x_train, positive_x, negative_x], ['orig', 'pos', 'neg']):
    #    visualize_sample(data, name)

    model.train(positive_x, negative_x)

    print('train error:', 1.0 - model.predict(x_train).eq(y_train).float().mean().item())

    x_test, y_test = next(iter(test_loader))
    x_test, y_test = x_test.to(device), y_test.to(device)

    print('test error:', 1.0 - model.predict(x_test).eq(y_test).float().mean().item())
