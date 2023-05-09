import torch
from networks import FFDenseNet
from layers import FFLayer
from functions import embed_label
from utils import mnist_loaders

TRAIN_BATCH = 50_000
TEST_BATCH = 10_000
MNIST_NORMS = ((0.1307,), (0.3081,))
IN_DIM = 784
HIDDEN_DIM = 500
OUT_DIM = 500


if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cpu"  # Update to use cuda if available
    train_loader, test_loader = mnist_loaders(TRAIN_BATCH, TEST_BATCH, MNIST_NORMS)

    model = FFDenseNet([IN_DIM, HIDDEN_DIM, OUT_DIM], layer=FFLayer, device=device)
    x_train, y_train = next(iter(train_loader))
    x_train, y_train = x_train.to(device), y_train.to(device)
    positive_x = embed_label(x_train, y_train)
    random_label = torch.randperm(x_train.size(0))
    negative_x = embed_label(x_train, y_train[random_label])

    model.train(positive_x, negative_x)

    print('train error:', 1.0 - model.predict(x_train).eq(y_train).float().mean().item())

    x_test, y_test = next(iter(test_loader))
    x_test, y_test = x_test.to(device), y_test.to(device)

    print('test error:', 1.0 - model.predict(x_test).eq(y_test).float().mean().item())
