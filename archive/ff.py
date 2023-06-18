from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from typing import Callable, Tuple


########################################################
########################################################
############ FF ALGORITHM UTILITY FUNCTIONS ############
########################################################
########################################################

# DATASET PROCESSING -- COPIED OVER
class FF_Dataset(torch.utils.data.Dataset):

    def __init__(self,
                 dataloader: DataLoader):
        self.dataset = [batch for batch_pos, batch_neg in dataloader
                        for batch in zip(batch_pos, batch_neg)]

    def __getitem__(self,
                    index: int):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


# ADD LABEL TO DATAPOINTS -- COPIED OVER
def add_labels_to_data_fn(X: torch.Tensor, Y: torch.Tensor, only_positive: bool):
    """Generate positive and negative samples using labels. It overlays labels in input. For neg it does
    the same but with shuffled labels.
    Args:
        X (torch.Tensor): batch of samples
        Y (torch.Tensor): batch of labels
        only_positive (bool): if True, it outputs only positive exmples with labels overlayed
    Returns:
        Tuple[torch.Tensor]: batch of positive (and negative samples)
    """
    X_pos = X.clone()

    X_pos[:, :10] *= 0.0
    X_pos[range(X.shape[0]), Y] = X_pos.max()  # one hot

    if only_positive:
        return X_pos
    else:
        X_neg = X.clone()
        rnd = torch.randperm(X_neg.size(0))
        # Y_neg = (Y + torch.randint(1, (Y.max()-1), (Y.shape[0],))) % Y.max() # still don't get why does not work
        Y_neg = Y[rnd]
        X_neg[:, :10] *= 0.0
        X_neg[range(X_neg.shape[0]), Y_neg] = X_neg.max()  # one hot

        return X_pos, X_neg


# GOODNESS FUNCTION
def hinton_goodness(x):
    return x.pow(2).mean(1)


# LOSS FUNCTION -- COPIED OVER
def default_loss(X_pos: torch.Tensor, X_neg: torch.Tensor, th: float):
    """Base loss described in the paper. Log(1+exp(x)) is added to help differentiation.
    Args:
        X_pos (torch.Tensor): batch of positive model predictions
        X_neg (torch.Tensor): batch of negative model predictions
        th (float): loss function threshold
    Returns:
        torch.Tensor: output loss
    """
    logits_pos = X_pos.pow(2).mean(dim=1)
    logits_neg = X_neg.pow(2).mean(dim=1)

    loss_pos = - logits_pos + th
    loss_neg = logits_neg - th

    loss_poss = torch.log(1 + torch.exp(loss_pos)).mean()
    loss_neg = torch.log(1 + torch.exp(loss_neg)).mean()

    loss = loss_poss + loss_neg

    return loss


# EVALUATING MODEL RESULTS
def FF_test(model, device, test_loader):
    for test_X, test_Y in test_loader:
        test_X = test_X.to(device)
        test_Y = test_Y.to(device)

        acc += (model.predict(test_X, add_labels_to_data_fn, 10).eq(test_Y).sum())

    print('\nTest Accuracy: {}/{} ({:.0f}%)\n'.format(100. * acc / len(test_loader.dataset)))


########################################################
########################################################
############# DEFINING  MLP ARCHITECTURES ##############
########################################################
########################################################

# GENERIC MLP MODEL -- NOT WITH FF LAYERS
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 10)
        # self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        # x = F.relu(x)
        # x = self.dropout2(x)
        # x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


# MLP MODEL -- !!!!!WITH!!!! FF LAYERS
class FF_MLP(nn.Module):  # , FF_Model_Skeleton):
    def __init__(self):
        super(FF_MLP, self).__init__()
        self.conv1 = FF_Conv2D(1, 32, 3, 1)
        self.conv2 = FF_Conv2D(32, 64, 3, 1)
        self.fc1 = FF_Linear(9216, 10, nn.functional.relu, 10)
        self.layers = [self.conv1, self.conv2, self.fc1]

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)
        return output

    def predict(self,
                X: torch.tensor,
                label_fn: Callable,
                num_classes: int):

        class_wise_goodnesses = []
        for class_idx in range(num_classes):
            goodness = []
            output = label_fn(X, class_idx)
            for layer in self.layers:
                goodness += [hinton_goodness(layer(output))]

            total_goodness = sum(goodness)
            class_wise_goodnesses += [total_goodness.unsqueeze(1)]

        best_class = torch.cat(class_wise_goodnesses, 1).argmax(1)
        return best_class

    def get_layer_losses(self,
                         batch_pos: torch.Tensor,
                         batch_neg: torch.Tensor):

        layer_losses = []

        for layer in self.layers:
            batch_pos, batch_neg, loss = layer.train_layer(batch_pos, batch_neg)
            layer_losses.append(loss)

        return layer_losses

    def train_layers(self,
                     epochs: int,
                     dataloader: DataLoader):

        # Train each layer and then update the dataset to be the output of that layer for input to training the next layer
        for i in range(len(self.layers)):
            cur_layer = self.layers[i]

            for cur_ep in range(epochs):
                for batch_pos, batch_neg in train_dataloader:
                    pos_out, neg_out, loss = layer.train_layer(batch_pos, batch_neg)
                print("Layer", i + 1, "Epoch", cur_ep, "Loss:", loss)

            batch_size = dataloader.batch_size
            updated_dataset = FF_Dataset(
                (cur_layer(batch_pos), cur_layer(batch_neg)) for batch_pos, batch_neg in dataloader)

            train_dataloader = torch.utils.data.DataLoader(updated_dataset, batch_size=batch_size, shuffle=True)


# FF LINEAR LAYER -- HAS ITS OWN TRAIN FUNCTION
class FF_Linear(nn.Linear):

    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 layer_activation: torch.nn,
                 layer_optimizer: torch.optim,
                 layer_LR: float,  # learning rate
                 layer_loss_threshold: float,
                 layer_loss_fn: Callable,
                 bias: bool = True):
        super(FF_Linear, self).__init__(in_dim, out_dim, bias)

        self.layer_activation = layer_activation
        self.layer_optimizer = layer_optimizer(self.parameters(), lr=layer_LR)
        self.layer_loss_threshold = layer_loss_threshold
        self.layer_loss_fn = layer_loss_fn

    def forward(self,
                X: torch.Tensor):
        # Normalize Input
        X = X / (X.norm(2, 1, keepdim=True))

        # Return f(W*X + b)
        return self.layer_activation(
            torch.mm(X, self.weight.T) + self.bias.unsqueeze(0))  # self.weight and self.bias part of nn.Linear

    def train_layer(self,
                    batch_pos: torch.Tensor,
                    batch_neg: torch.Tensor):
        # Get output activations for positive and negative samples
        batch_pos_output = self.forward(batch_pos)
        batch_neg_output = self.forward(batch_neg)

        # Compute layer loss and do local (i.e. layer-level) computation of gradient
        loss = self.layer_loss_fn(batch_pos_output, batch_neg_output, self.layer_loss_threshold)
        self.layer_optimizer.zero_grad()
        loss.backward()
        self.layer_optimizer.step()

        return batch_pos_output.detach(), batch_neg_output.detach(), loss.item()


# FF CONV2D LAYER -- HAS ITS OWN TRAIN FUNCTION
class FF_Conv2D(nn.Conv2d):

    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 kernel_size: int,
                 layer_activation: torch.nn,
                 layer_optimizer: torch.optim,
                 layer_LR: float,  # learning rate
                 layer_loss_threshold: float,
                 layer_loss_fn: Callable,
                 bias: bool = True):
        super(FF_Conv2D, self).__init__(in_dim, out_dim, kernel_size)

        self.layer_activation = layer_activation
        self.layer_optimizer = layer_optimizer(self.parameters(), lr=layer_LR)
        self.layer_loss_threshold = layer_loss_threshold
        self.layer_loss_fn = layer_loss_fn

    def forward(self,
                X: torch.Tensor):
        # Normalize Input
        X = X / (X.norm(2, 1, keepdim=True))

        return F.conv2d(X, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def train_layer(self,
                    batch_pos: torch.Tensor,
                    batch_neg: torch.Tensor):
        # Get output activations for positive and negative samples
        batch_pos_output = self.forward(batch_pos)
        batch_neg_output = self.forward(batch_neg)

        # Compute layer loss and do local (i.e. layer-level) computation of gradient
        loss = self.layer_loss_fn(batch_pos_output, batch_neg_output, self.layer_loss_threshold)
        self.layer_optimizer.zero_grad()
        loss.backward()
        self.layer_optimizer.step()

        return batch_pos_output.detach(), batch_neg_output.detach(), loss.item()


########################################################
########################################################
###################### MAIN CODE  ######################
########################################################
########################################################
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_data = datasets.MNIST('../data', train=True, download=True,
                                transform=transform)
    test_data = datasets.MNIST('../data', train=False,
                               transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, **test_kwargs)

    # Overlay data with labels for supervised learning with FF
    train_data_labeled = FF_Dataset(add_labels_to_data_fn(X.to(device), Y.to(device), False) for X, Y in train_loader)

    FF_train_loader = torch.utils.dataset.DataLoader(train_data_labeled, batch_size=train_loader.batch_size,
                                                     shuffle=True)

    ############################################################
    ############### CREATE THE MODEL AND TRAIN IT ##############
    ############################################################

    model = FF_MLP().to(device)
    model.train_layers(epochs=args.epochs, dataloader=FF_train_loader)

    # Test the model
    FF_test(model, device, test_loader)

    # activation = torch.nn.ReLU()
    # layer_LR = 0.09
    # layer_optimizer = torch.optim.Adam
    # layer_loss_threshold = 9.0
    # layer_loss_fn = default_loss
    # optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    # for epoch in range(1, args.epochs + 1):
    #     train(args, model, device, train_loader, optimizer, epoch)
    #     test(model, device, test_loader)
    #     scheduler.step()

    # if args.save_model:
    #     torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()


########################################################
########################################################
###################### OLD CODE ########################
########################################################
########################################################

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
