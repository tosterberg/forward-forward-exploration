import numpy as np
import torch
from src import utils
from abc import ABC, abstractmethod

class Classifier(ABC):
    def __init__(self, opt, num_classes):
        self.opt = opt
        self.num_classes = num_classes

    def __getitem__(self, index):
        sample, class_label = self._generate_sample(
            index
        )

        inputs = {
            "sample": sample
        }
        labels = {"class_labels": class_label}
        return inputs, labels

    @staticmethod
    def _get_samples(sample, class_label):
        return sample, class_label

    @abstractmethod
    def _generate_sample(self, index):
        pass


class MNIST(Classifier, torch.utils.data.Dataset):
    def __init__(self, opt, partition, num_classes=10):
        super().__init__(opt, num_classes)
        self.data = utils.get_MNIST_partition(self.opt, partition)

    def __len__(self):
        return len(self.data)

    def _generate_sample(self, index):
        # Get MNIST sample.
        sample, class_label = self.data[index]
        return self._get_samples(sample, class_label)


class CIFAR10(Classifier, torch.utils.data.Dataset):
    def __init__(self, opt, partition, num_classes=10):
        super().__init__(opt, num_classes)
        self.data = utils.get_CIFAR10_partition(opt, partition)

    def __len__(self):
        return len(self.data)

    def _generate_sample(self, index):
        # Get CIFAR sample.
        sample, class_label = self.data[index]
        return self._get_samples(sample, class_label)
