import pickle
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

N_PARTITIONS = 3


def _transform(x_data: np.ndarray) -> np.ndarray:
    """Feature transformation function, preserving length of data."""
    return x_data.reshape((len(x_data), -1)) / 255.0


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


def get_mnist_data() -> tuple[
    tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]
]:
    """Get test data for MNIST dataset.

    Returns:
        tuple containing tran and test data as (featrues, labels)
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # TODO: For now we reshape to flatted features (no 2D images)
    return (_transform(x_train), y_train), (_transform(x_test), y_test)


def get_mnist_datafiles(n_partitions=N_PARTITIONS) -> list[Path]:
    """Get paths to MNIST datafiles for training.

    Args:
        n_partitions: number of datasets to create

    Returns:
        list of paths to created dataset files
    """
    folder = Path("./output/data")
    folder.mkdir(parents=True, exist_ok=True)

    (x_train, y_train), _ = get_mnist_data()
    return create_datasets(x_train, y_train, folder, n_partitions)


if __name__ == "__main__":
    get_mnist_datafiles(N_PARTITIONS)
