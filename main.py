import time
from collections import defaultdict
import hydra
import os
import json
import yaml
import torch
from omegaconf import DictConfig
from src import utils

test_record = {
    "test_results": [],
    "train_times": [],
    "inference_times": []
}

def reset_test_record():
    global test_record
    test_record= None
    test_record = {
        "test_results": [],
        "train_times": [],
        "inference_times": []
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
        test_record["train_times"].append(time.time() - start_time)
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
            test_record["inference_times"].append(time.time() - test_time)

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


def full_test_run():
    overwrites = ["dataset", "type", "hidden_dim", "num_layers", "threshold"]
    bpe_options = {
        "input": {
            "dataset": ["mnist", "cifar10"],
        } ,
        "model": {
            "name": "model",
            "type": "linear",
            "hidden_dim": [1000, 2000, 4000],
            "num_layers": [2, 4],
        },
        "training": {
            "threshold": 0
        }
    }

    cnn_options = {
        "input": {
            "dataset": ["mnist", "cifar10"],
        },
        "model": {
            "name": "model",
            "type": "mlp",
            "hidden_dim": 4,
            "num_layers": 4,
        },
        "training": {
            "threshold": 0
        }
    }

    ff_options = {
        "input": {
            "dataset": ["mnist", "cifar10"],
        },
        "model": {
            "name": "ffmodel",
            "type": "linear",
            "hidden_dim": [1000, 2000, 4000],
            "num_layers": [2, 4],
        },
        "training": {
            "threshold": [0.5]
        }
    }

    bpe_sequences = utils.flatten_object_on_keys(bpe_options, overwrites)
    cnn_sequences = utils.flatten_object_on_keys(cnn_options, overwrites)
    ff_sequences = utils.flatten_object_on_keys(ff_options, overwrites)
    test_sequences = bpe_sequences + ff_sequences + cnn_sequences

    for idx, test in enumerate(test_sequences):
        print(f'Starting test {idx} of {len(test_sequences)}')
        reset_test_record()
        f = open("config.yaml", "r")
        config = yaml.safe_load(f)
        f.close()
        config["input"]["dataset"] = test["input"]["dataset"]
        config["model"]["name"] = test["model"]["name"]
        config["model"]["type"] = test["model"]["type"]
        config["model"]["hidden_dim"] = test["model"]["hidden_dim"]
        config["model"]["num_layers"] = test["model"]["num_layers"]
        config["training"]["threshold"] = test["training"]["threshold"]
        with open("config.yaml", "w") as f:
            yaml.dump(config, f)
            f.close()
        run()
        print(f"Test complete: {test}")
        print(f'Finished test {idx} of {len(test_sequences)}')
        print("============================================")


def analysis():
    paths = []
    for root, dirs, files in os.walk(r"tests/"):
        for file in files:
            if file.endswith(".json"):
                inf = None
                trn = None
                loss = []
                acc = []
                ssq = []
                f = open(os.path.join(root, file))
                json_data = json.load(f)
                for key in json_data:
                    if key == "inference_times":
                        inf = json_data[key]
                    elif key == "train_times":
                        trn = json_data[key]
                    else:
                        for item in json_data[key]:
                            for k in item:
                                if k == "Loss":
                                    loss.append(item[k])
                                elif k == "classification_accuracy":
                                    acc.append(item[k])
                                elif k == "multi_pass_classification_accuracy":
                                    ssq.append(item[k])

                record = {
                    "path": os.path.join(root, file),
                    "title": file,
                    "inference_times": inf,
                    "training_times": trn,
                    "loss": loss.copy(),
                    "acc": acc.copy(),
                    "ssq": ssq.copy()
                }
                paths.append(record)
                f.close()

    for path in paths:
        print("\n")
        print(path['title'])
        print("============================")
        print(f'Average Training Time:  {sum(path["training_times"])/len(path["training_times"])}')
        print(f'Average Inference Time: {sum(path["inference_times"])/len(path["inference_times"])}')
        if len(path["ssq"]) > 0:
            print(f'Max SSQ Acc:            {max(path["ssq"])}')
        print(f'Max Acc:                {max(path["acc"])}')
        print(f'Min Loss:               {min(path["loss"])}')


if __name__ == "__main__":
    full_test_run()
    analysis()
