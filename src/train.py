"""Test training process."""
import json
import random
from pathlib import Path

import numpy as np
from data import get_mnist_data, get_mnist_datafiles
from feltlabs.algorithm import aggregate, train
from feltlabs.model import load_model
from nacl.public import PrivateKey
from sklearn.metrics import accuracy_score

aggregation_key = PrivateKey.generate()
scientist_key = PrivateKey.generate()

model_def = {
    "model_type": "sklearn",
    "model_name": "Lasso",
}


def test_training(n_partitions: int = 3):
    # Prepare folders to use
    folder = Path("output")

    input_folder = folder / "input" / "fake_did"
    output_folder = folder / "models" / "fake_did"
    output_final = folder / "final_model"

    input_folder.mkdir(parents=True, exist_ok=True)
    output_folder.mkdir(parents=True, exist_ok=True)
    output_final.mkdir(parents=True, exist_ok=True)

    # Create custom data file (containing model definition)
    with open(input_folder.parent / "algoCustomData.json", "w") as f:
        json.dump(model_def, f)

    ### Training section ###
    args_str = f"--output_folder {output_folder}"
    args_str += f" --input_folder {input_folder.parent}"
    args_str += f" --data_type pickle"
    args_str += f" --aggregation_key {bytes(aggregation_key.public_key).hex()}"

    seeds = []
    files = get_mnist_datafiles()

    for i, file_path in enumerate(files):
        # Move file to input folder as train train dataset
        file_path.rename(input_folder / "0")

        seeds.append(random.randint(0, 1000000))

        args_str_final = f"{args_str} --seed {seeds[-1]}"
        train.main(args_str_final.split(), f"{i}")

    ### Aggregation section ###
    args_str = f"--output_folder {output_final}"
    args_str += f" --input_folder {output_folder.parent}"
    args_str += f" --private_key {bytes(aggregation_key).hex()}"

    aggregate.main(args_str.split(), "model")

    ### Test final results ###
    final_model = load_model(output_final / "model")
    final_model.remove_noise_models(seeds)

    (x_train, y_train), (x_test, y_test) = get_mnist_data()

    def model_test(model, name=""):
        y_pred = model.predict(x_test)
        y_pred = np.rint(y_pred)

        print(f"\nModel - {name}")
        print(y_pred)
        print("Accuracy", accuracy_score(y_test, y_pred))

    model_test(final_model, "aggregated")

    ### Fully trained model ###
    new_model = load_model(input_folder.parent / "algoCustomData.json")
    new_model.fit(x_train, y_train)

    model_test(new_model, "full_data")


if __name__ == "__main__":
    test_training()
