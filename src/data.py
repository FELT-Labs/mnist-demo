import pickle
import random
import shutil
from pathlib import Path
from typing import Callable

import numpy as np
import tensorflow as tf


def transform_flat(x_data: np.ndarray) -> np.ndarray:
    """Feature transformation function, preserving length of data."""
    return x_data.reshape((len(x_data), -1)) / 255.0


def transform_cnn(x_data: np.ndarray) -> np.ndarray:
    """Feature transformation function, for CNN networks requiring (32, 32, 3) shape."""
    data = (x_data.reshape((len(x_data), 28, 28, 1)) - 127.5) / 127.5
    return np.pad(np.concatenate(3 * [data], axis=-1), [(0, 0), (2, 2), (2, 2), (0, 0)])


def create_datasets(
    x_train: np.ndarray, y_train: np.ndarray, output_folder: Path, n_partitions: int = 3
) -> list[Path]:
    """Create random partitions of original data and store them in dataset files.

    Args:
        x_train: training data - features
        y_train: training data - labels
        output_folder: destination to store the data files
        n_partitions: number of separate datasets to create

    Returns:
        list of paths to created dataset files
    """
    # Make it reproducible
    random.seed(42)

    index_list = list(range(len(x_train)))
    random.shuffle(index_list)

    partitions = [index_list[i::n_partitions] for i in range(n_partitions)]

    files = []
    for i, indexes in enumerate(partitions):
        data_tuple = (x_train[indexes], y_train[indexes])

        path = output_folder / f"{i}.pkl"
        with open(path, "wb") as f:
            pickle.dump(data_tuple, f)

        files.append(path)

    return files


def get_mnist_data(
    transform: Callable[[np.ndarray], np.ndarray]
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Get test data for MNIST dataset.

    Returns:
        tuple containing tran and test data as (featrues, labels)
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    return (transform(x_train), y_train), (transform(x_test), y_test)


def get_mnist_datafiles(
    transform: Callable[[np.ndarray], np.ndarray], n_partitions=3
) -> list[Path]:
    """Get paths to MNIST datafiles for training.

    Args:
        n_partitions: number of datasets to create

    Returns:
        list of paths to created dataset files
    """
    folder = Path("./output_data")
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)

    (x_train, y_train), _ = get_mnist_data(transform)
    return create_datasets(x_train, y_train, folder, n_partitions)


if __name__ == "__main__":
    # TODO: Argparse - for generating data using CLI
    # TODO: Add FELT json config file as well
    partitions = 3
    get_mnist_datafiles(transform_flat, partitions)
