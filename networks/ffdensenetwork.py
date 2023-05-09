import torch
from layers import FFLayer
from functions import embed_label


class FFDenseNet(torch.nn.Module):
    def __init__(self, dims, layer=FFLayer, device="cpu"):
        super().__init__()
        self.layers = []
        for d in range(len(dims) - 1):
            self.layers += [layer(dims[d], dims[d + 1]).to(device)]

    def predict(self, x):
        goodness_per_label = []
        for label in range(10):
            h = embed_label(x, label)
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness += [h.pow(2).mean(1)]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(1)

    def train(self, positive_input, negative_input):
        hidden_positive, hidden_negative = positive_input, negative_input
        for i, layer in enumerate(self.layers):
            hidden_positive, hidden_negative = layer.train(hidden_positive, hidden_negative)
