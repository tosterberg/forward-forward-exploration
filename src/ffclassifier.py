import numpy as np
import torch
from src import utils
from abc import ABC, abstractmethod

class FFClassifier(ABC):
    def __init__(self, opt, num_classes):
        self.opt = opt
        self.num_classes = num_classes
        self.uniform_label = torch.ones(self.num_classes) / self.num_classes

    def __getitem__(self, index):
        pos_sample, neg_sample, neutral_sample, all_sample, class_label = self._generate_sample(
            index
        )

        inputs = {
            "pos_images": pos_sample,
            "neg_images": neg_sample,
            "neutral_sample": neutral_sample,
            "all_sample": all_sample
        }
        labels = {"class_labels": class_label}
        return inputs, labels

    def _get_pos_sample(self, sample, class_label):
        one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(class_label).long(), num_classes=self.num_classes
        )
        pos_sample = sample.clone()
        pos_sample[0, 0, : self.num_classes] = one_hot_label
        return pos_sample

    def _get_neg_sample(self, sample, class_label):
        classes = list(range(self.num_classes))
        classes.remove(class_label)  # Remove true label from possible choices.
        wrong_class_label = np.random.choice(classes)
        one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(wrong_class_label).long(), num_classes=self.num_classes
        )
        neg_sample = sample.clone()
        neg_sample[0, 0, : self.num_classes] = one_hot_label
        return neg_sample

    def _get_neutral_sample(self, sample):
        sample[0, 0, : self.num_classes] = self.uniform_label
        return sample

    def _get_all_sample(self, sample):
        all_samples = torch.zeros((self.num_classes, sample.shape[0], sample.shape[1], sample.shape[2]))
        for i in range(self.num_classes):
            all_samples[i, :, :, :] = sample.clone()
            one_hot_label = torch.nn.functional.one_hot(
                torch.tensor(i).long(), num_classes=self.num_classes)
            all_samples[i, 0, 0, : self.num_classes] = one_hot_label.clone()
        return all_samples

    def _get_samples(self, sample, class_label):
        pos_sample = self._get_pos_sample(sample, class_label)
        neg_sample = self._get_neg_sample(sample, class_label)
        neutral_sample = self._get_neutral_sample(sample)
        all_sample = self._get_all_sample(sample)
        return pos_sample, neg_sample, neutral_sample, all_sample, class_label

    @abstractmethod
    def _generate_sample(self, index):
        pass


class FF_MNIST(FFClassifier, torch.utils.data.Dataset):
    def __init__(self, opt, partition, num_classes=10):
        super().__init__(opt, num_classes)
        self.data = utils.get_MNIST_partition(self.opt, partition)

    def __len__(self):
        return len(self.data)

    def _generate_sample(self, index):
        # Get MNIST sample.
        sample, class_label = self.data[index]
        return self._get_samples(sample, class_label)


class FF_CIFAR10(FFClassifier, torch.utils.data.Dataset):
    def __init__(self, opt, partition, num_classes=10):
        super().__init__(opt, num_classes)
        self.data = utils.get_CIFAR10_partition(opt, partition)

    def __len__(self):
        return len(self.data)

    def _generate_sample(self, index):
        # Get CIFAR sample.
        sample, class_label = self.data[index]
        return self._get_samples(sample, class_label)
