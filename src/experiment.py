"""Test training process."""
from typing import Any, Callable

import numpy as np
from feltlabs.model import load_model
from sklearn.metrics import accuracy_score

from data import get_mnist_data, get_mnist_datafiles, transform_cnn, transform_flat
from train import federated_training

experiments = [
    {
        "iterations": 1,
        "full_iterations": 1,
        "partitions": 3,
        "data_transform": transform_flat,
        "model_definition": {
            "model_type": "sklearn",
            "model_name": "LogisticRegression",
            "init_params": {"max_iter": 100},
        },
    },
    {
        "iterations": 1,
        "full_iterations": 1,
        "partitions": 3,
        "data_transform": transform_flat,
        "model_definition": {
            "model_type": "sklearn",
            "model_name": "NearestCentroidClassifier",
            "init_params": {},
        },
    },
    {
        "iterations": 8,
        "full_iterations": 8,
        "partitions": 3,
        "data_transform": transform_flat,
        "model_definition": {
            "model_type": "sklearn",
            "model_name": "MLPClassifier",
            "init_params": {
                "hidden_layer_sizes": [50, 50],
                "max_iter": 50,
                "warm_start": True,
                "random_state": 10,
            },
        },
    },
    {
        "iterations": 6,
        "full_iterations": 6,
        "partitions": 3,
        "data_transform": transform_cnn,
        "model_definition": {
            "model_type": "tensorflow",
            "model_name": "EfficientNetB0",
            "init_params": {
                "input_shape": [32, 32, 3],
                "classes": 10,
                "include_top": True,
                "weights": None,
            },
        },
        "fit_args": {"epochs": 1, "batch_size": 32},
    },
    {
        "iterations": 10,
        "full_iterations": 10,
        "partitions": 3,
        "data_transform": lambda data: data.reshape((len(data), 28, 28, 1)) / 255.5,
        "model_definition": {
            "model_type": "tensorflow",
            "model_name": "CustomCNN",
            "init_params": {
                "input_shape": [28, 28, 1],
                "classes": 10,
                "architecture": "C-20-3-2-same,M-3-2,F,H-100",
            },
        },
        "fit_args": {"epochs": 1, "batch_size": 32},
    },
]


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
    new_model = load_model(model_definition)
    for _ in range(full_iterations):
        new_model.fit(x_train, y_train)
    full_accuracy = eval_function(new_model)

    return train_results, full_accuracy


if __name__ == "__main__":
    train_acc, full_acc = run_experiment(**experiments[0])
    print("Results")
    print(train_acc)
    print(full_acc)
