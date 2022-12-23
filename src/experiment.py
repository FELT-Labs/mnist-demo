"""Test training process."""
from typing import Any, Callable

import numpy as np
from feltlabs.model import load_model
from sklearn.metrics import accuracy_score

from data import get_mnist_data, get_mnist_datafiles
from experiments import experiments
from train import federated_training


def _create_eval_function(x_test, y_test):
    """Create evaluation function for the models."""

    def model_test(model):
        y_pred = model.predict(x_test)
        # tensorflow model uses softmax, need to reduce
        if len(y_pred.shape) == 2:
            y_pred = np.argmax(y_pred, axis=-1)
        else:
            y_pred = np.rint(y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        print(accuracy)
        return accuracy

    return model_test


def run_experiment(
    model_definition: dict,
    data_transform: Callable,
    partitions: int,
    iterations: int,
    full_iterations: int,
):
    """Run provided experiment and evaluate the models.

    Args:
        model_definition: dictionary defining the model to use
        data_transform: function taking data (np.array) and preparing them for training
        partitions: number of partitions created from original data
        iterations: number of federated learning iterations
        full_iterations: number of iterations used for centralized training

    Returns:
        dictionary with training accuracies from federated training
        and accuracy from centralized training
    """

    (x_train, y_train), (x_test, y_test) = get_mnist_data(data_transform)
    eval_function = _create_eval_function(x_test, y_test)

    files = get_mnist_datafiles(data_transform, partitions)
    # Train local models on data files and get evaluation results
    _, train_results = federated_training(
        model_definition, files, eval_function, iterations
    )

    # Train model on all centralized data
    if model_definition["model_type"] == "sklearn":
        model_definition["init_params"]["max_iter"] = 200
    new_model = load_model({"model_definition": model_definition})
    for _ in range(full_iterations):
        new_model.fit(x_train, y_train)
    full_accuracy = eval_function(new_model)

    return train_results, full_accuracy


if __name__ == "__main__":
    train_acc, full_acc = run_experiment(**experiments[0])
    print("Results")
    print(train_acc)
    print(full_acc)
