import time
from collections import defaultdict
import os
import argparse
import json
import matplotlib.pyplot as plt
import hydra
import torch
from omegaconf import DictConfig
from src import utils

test_record = {
    "test_results": []
}

def train(opt, model, optimizer):
    start_time = time.time()
    train_loader = utils.get_data(opt, "train")
    num_steps_per_epoch = len(train_loader)
    best_val_acc = 0.0

    for epoch in range(opt.training.epochs):
        train_results = defaultdict(float)
        optimizer = utils.update_learning_rate(optimizer, opt, epoch)

        for inputs, labels in train_loader:
            inputs, labels = utils.preprocess_inputs(opt, inputs, labels) # push to GPU
            optimizer.zero_grad()

            scalar_outputs = model(inputs, labels)
            scalar_outputs["Loss"].backward()
            optimizer.step()

            train_results = utils.log_results(
                train_results, scalar_outputs, num_steps_per_epoch
            )

        utils.print_results("train", time.time() - start_time, train_results, epoch)
        start_time = time.time()

        # Validate.
        if epoch % opt.training.val_idx == 0 and opt.training.val_idx != -1:
            best_val_acc = validate_or_test(opt, model, "val", epoch=epoch, best_val_acc=best_val_acc)

    return model


def validate_or_test(opt, model, partition, epoch=None, best_val_acc=1.0):
    test_time = time.time()
    test_results = defaultdict(float)

    data_loader = utils.get_data(opt, partition)
    num_steps_per_epoch = len(data_loader)

    model.eval()
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = utils.preprocess_inputs(opt, inputs, labels)

            scalar_outputs = model.forward_downstream_classification_model(
                inputs, labels
            )
            if opt.model.name == "ffmodel":
                scalar_outputs = model.forward_downstream_ssq_classification(
                    inputs, labels, scalar_outputs=scalar_outputs
                )
            test_results = utils.log_results(
                test_results, scalar_outputs, num_steps_per_epoch
            )
            test_record["test_results"].append(test_results)

    utils.print_results(partition, time.time() - test_time, test_results, epoch=epoch)
    # save model if classification accuracy is better than previous best
    if test_results["classification_accuracy"] > best_val_acc:
        print("saving model")
        best_val_acc = test_results["classification_accuracy"]
        utils.save_model(model, opt)

    model.train()
    return best_val_acc


@hydra.main(config_path=".", config_name="config", version_base=None)
def run(opt: DictConfig) -> None:
    opt = utils.parse_args(opt)
    model, optimizer = utils.get_model_and_optimizer(opt)
    model = train(opt, model, optimizer)
    validate_or_test(opt, model, "val")

    utils.save_record(test_record, opt)

    if opt.training.final_test:
        validate_or_test(opt, model, "test")


if __name__ == "__main__":
    run()
