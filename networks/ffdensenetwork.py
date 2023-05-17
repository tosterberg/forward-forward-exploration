import torch
import math
import utils
from layers import FFLayer
from functions import embed_label


class FFDenseNet(torch.nn.Module):
    def __init__(self, dims, layer=FFLayer, device="cpu", layer_config=None):
        super().__init__()
        if layer_config is None:
            layer_config = {}
        self.layers = []
        for d in range(len(dims) - 1):
            self.layers += [layer(dims[d], dims[d + 1], device=device, **layer_config).to(device)]

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


class FFClassificationModel(torch.nn.Module):

    def __init__(self, config):
        super(FFClassificationModel, self).__init__()
        self.config = config
        self.channels = [self.config.model.hidden_dim] * self.config.model.num_layers
        self.activation_func = torch.nn.ReLU()
        self.input_layer_size = utils.get_input_layer_size(config)

        self.model = torch.nn.ModuleList([torch.nn.Linear(self.input_layer_size, self.channels[0])])
        for i in range(1, len(self.channels)):
            self.model.append(torch.nn.Linear(self.channels[i-1], self.channels[i]))

        self.loss = torch.nn.BCEWithLogitsLoss()

        self.means = [
            torch.zeros(self.channels[i], device=self.config.device) + 0.5
            for i in range(self.config.model.num_layers)
        ]

        self.channels_classification_loss = sum(
            self.channels[-i] for i in range(self.config.model.num_layers - 1)
        )
        self.linear_classifier = torch.nn.Sequential(
            torch.nn.Linear(self.channels_classification_loss, 10, bias=False)
        )
        self.classification_loss = torch.nn.CrossEntropyLoss()
        self._initialize()

    def _initialize(self):
        for m in self.model.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(
                    m.weight, mean=0, std=1 / math.sqrt(m.weight.shape[0])
                )
                torch.nn.init.zeros_(m.bias)

        for m in self.linear_classifier.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.zeros_(m.weight)

    def _layer_norm(self, z, eps=1e-8):
        return z / (torch.sqrt(torch.mean(z ** 2, dim=-1, keepdim=True)) + eps)

    def _calc_peer_normalization_loss(self, idx, z):  # z is bs*2, 2000
        # Only calculate mean activity over positive samples.
        mean_activity = torch.mean(z[:self.opt.input.batch_size], dim=0)  # bsx2000 -> 2000

        self.running_means[idx] = self.running_means[
                                      idx
                                  ].detach() * self.opt.model.momentum + mean_activity * (
                                          1 - self.opt.model.momentum
                                  )

        peer_loss = (torch.mean(self.running_means[idx]) - self.running_means[idx]) ** 2
        return torch.mean(peer_loss)

    def _calc_ff_loss(self, z, labels):
        sum_of_squares = torch.sum(z ** 2, dim=-1)  # sum of squares of each activation. bs*2
        # s - thresh    --> sigmoid --> cross entropy

        logits = sum_of_squares - z.shape[1]  # if the average value of each activation is >1, logit is +ve, else -ve.
        ff_loss = self.ff_loss(logits,
                               labels.float())  # labels are 0 or 1, so convert to float. logits->sigmoid->normal cross entropy

        with torch.no_grad():
            ff_accuracy = (
                    torch.sum(
                        (torch.sigmoid(logits) > 0.5) == labels)  # threshold is logits=0, so sum of squares = 784
                    / z.shape[0]
            ).item()
        return ff_loss, ff_accuracy

    def forward(self, inputs, labels):
        scalar_outputs = {
            "Loss": torch.zeros(1, device=self.opt.device),
            "Peer Normalization": torch.zeros(1, device=self.opt.device),
        }

        # Concatenate positive and negative samples and create corresponding labels.
        z = torch.cat([inputs["pos_images"], inputs["neg_images"]], dim=0)  # 2*bs, 1, 28, 28
        posneg_labels = torch.zeros(z.shape[0], device=self.opt.device)  # 2*bs
        posneg_labels[: self.opt.input.batch_size] = 1  # first BS samples true, next BS samples false

        z = z.reshape(z.shape[0], -1)  # 2*bs, 784
        z = self._layer_norm(z)

        for idx, layer in enumerate(self.model):
            z = layer(z)
            z = self.act_fn.apply(z)

            if self.opt.model.peer_normalization > 0:
                peer_loss = self._calc_peer_normalization_loss(idx, z)
                scalar_outputs["Peer Normalization"] += peer_loss
                scalar_outputs["Loss"] += self.opt.model.peer_normalization * peer_loss

            ff_loss, ff_accuracy = self._calc_ff_loss(z, posneg_labels)
            scalar_outputs[f"loss_layer_{idx}"] = ff_loss
            scalar_outputs[f"ff_accuracy_layer_{idx}"] = ff_accuracy
            scalar_outputs["Loss"] += ff_loss
            z = z.detach()

            z = self._layer_norm(z)

        scalar_outputs = self.forward_downstream_classification_model(
            inputs, labels, scalar_outputs=scalar_outputs
        )

        scalar_outputs = self.forward_downstream_multi_pass(
            inputs, labels, scalar_outputs=scalar_outputs
        )

        return scalar_outputs

    def forward_downstream_multi_pass(
            self, inputs, labels, scalar_outputs=None,
    ):
        if scalar_outputs is None:
            scalar_outputs = {
                "Loss": torch.zeros(1, device=self.opt.device),
            }

        z_all = inputs["all_sample"]  # bs, num_classes, C, H, W
        z_all = z_all.reshape(z_all.shape[0], z_all.shape[1], -1)  # bs, num_classes, C*H*W
        ssq_all = []
        for class_num in range(z_all.shape[1]):
            z = z_all[:, class_num, :]  # bs, C*H*W
            z = self._layer_norm(z)
            input_classification_model = []

            with torch.no_grad():
                for idx, layer in enumerate(self.model):
                    z = layer(z)
                    z = self.act_fn.apply(z)
                    z_unnorm = z.clone()
                    z = self._layer_norm(z)

                    if idx >= 1:
                        input_classification_model.append(z_unnorm)

            input_classification_model = torch.concat(input_classification_model,
                                                      dim=-1)  # bs x 6000 # concat all activations from all layers
            ssq = torch.sum(input_classification_model ** 2, dim=-1)  # bs # sum of squares of each activation
            ssq_all.append(ssq)
        ssq_all = torch.stack(ssq_all,
                              dim=-1)  # bs x num_classes # sum of squares of each activation for each class

        classification_accuracy = utils.get_accuracy(
            self.opt, ssq_all.data, labels["class_labels"]
        )

        scalar_outputs["multi_pass_classification_accuracy"] = classification_accuracy
        return scalar_outputs

    def forward_downstream_classification_model(
            self, inputs, labels, scalar_outputs=None,
    ):
        if scalar_outputs is None:
            scalar_outputs = {
                "Loss": torch.zeros(1, device=self.opt.device),
            }

        z = inputs["neutral_sample"]
        z = z.reshape(z.shape[0], -1)
        z = self._layer_norm(z)

        input_classification_model = []

        with torch.no_grad():
            for idx, layer in enumerate(self.model):
                z = layer(z)
                z = self.act_fn.apply(z)
                z = self._layer_norm(z)

                if idx >= 1:
                    input_classification_model.append(z)

        input_classification_model = torch.concat(input_classification_model,
                                                  dim=-1)  # concat all activations from all layers

        output = self.linear_classifier(input_classification_model.detach())  # bs x 10 ,
        output = output - torch.max(output, dim=-1, keepdim=True)[
            0]  # not entirely clear why each entry in output is made 0 or -ve
        classification_loss = self.classification_loss(output, labels["class_labels"])
        classification_accuracy = utils.get_accuracy(
            self.opt, output.data, labels["class_labels"]
        )

        scalar_outputs["Loss"] += classification_loss
        scalar_outputs["classification_loss"] = classification_loss
        scalar_outputs["classification_accuracy"] = classification_accuracy
        return scalar_outputs
