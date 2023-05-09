import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm
from torch.optim import Adam


class FFLayer(nn.Linear):
    def __init__(self, in_features, out_features,
                 bias=True, device=None, dtype=None,
                 optimizer=Adam, activation=torch.nn.ReLU(),
                 learning_rate=0.03, threshold=2.0, epochs=1000):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.activation = activation
        self.opt = optimizer(self.parameters(), lr=learning_rate)
        self.threshold = threshold
        self.num_epochs = epochs

    def forward(self, input: Tensor) -> Tensor:
        # TODO: make step value configurable
        x_d = input / (input.norm(2, 1, keepdim=True) + 1e-4)
        return self.activation(
            torch.mm(x_d, self.weight.T) + self.bias.unsqueeze(0)
        )

    # TODO: Try to make all forward passes a single forward call with a +/- signifier for
    #   imbalanced sampling of positive and negative data
    def train(self, positive_input, negative_input):
        for i in tqdm(range(self.num_epochs)):
            # TODO: Refactor to alternative goodness functions
            # Current function maximum goodness on positive input
            #   with minimum goodness on negative input
            #   Sum of Squares
            pos_goodness = self.forward(positive_input).pow(2).mean(1)
            neg_goodness = self.forward(negative_input).pow(2).mean(1)
            # TODO: Factor out to allow for different loss functions
            loss = torch.log(1 + torch.exp(torch.cat([
                -pos_goodness + self.threshold,
                neg_goodness - self.threshold]))).mean()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        return self.forward(positive_input).detach(), self.forward(negative_input).detach()
