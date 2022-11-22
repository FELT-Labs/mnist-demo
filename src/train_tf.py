"""Test training process."""
import json
import random
import shutil
from pathlib import Path

import numpy as np
from feltlabs.algorithm import aggregate, train
from feltlabs.config import AggregationConfig, TrainingConfig
from feltlabs.core.cryptography import decrypt_nacl
from feltlabs.model import load_model
from nacl.public import PrivateKey
from sklearn.metrics import accuracy_score

from data import get_mnist_data, get_mnist_datafiles, transform_cnn

N_PARTITIONS = 3
ITERATIONS = 1

aggregation_key = PrivateKey.generate()

model_def = {
    "model_type": "tensorflow",
    "model_name": "EfficientNetB0",
    "init_params": {
        "input_shape": [32, 32, 3],
        "classes": 10,
        "include_top": True,
        "weights": None,
    },
}

# model_def = {
#     "model_type": "sklearn",
#     "model_name": "MLPClassifier",
#     "init_params": {
#         "hidden_layer_sizes": [50, 50],
#         "max_iter": 50,
#         "warm_start": True,
#     },
# }


(x_train, y_train), (x_test, y_test) = get_mnist_data(transform_cnn)


def model_test(model, name=""):
    y_pred = np.argmax(
        model.predict(x_test),
        axis=-1,
    )

    print(f"\nModel - {name}")
    print(y_test)
    print(y_pred)
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy", acc)
    return acc


def test_training(n_partitions: int = 3):
    # Prepare folders to use
    folder = Path("output")
    if folder.exists():
        shutil.rmtree(folder)

    input_folder = folder / "input" / "fake_did"
    output_folder = folder / "models" / "fake_did"
    output_final = folder / "final_model"

    input_folder.mkdir(parents=True, exist_ok=True)
    output_folder.mkdir(parents=True, exist_ok=True)
    output_final.mkdir(parents=True, exist_ok=True)

    # Create custom data file (containing model definition)
    custom_data_path = input_folder.parent / "algoCustomData.json"
    with open(custom_data_path, "w") as f:
        json.dump(model_def, f)

    for i in range(ITERATIONS):
        ### Training section ###
        config = TrainingConfig(
            output_folder=output_folder,
            input_folder=input_folder.parent,
            custom_data_path=custom_data_path,
            data_type="pickle",
            experimental=True,
            aggregation_key=bytes(aggregation_key.public_key),
        )

        seeds = []
        local_models = []  # Only for testing
        files = get_mnist_datafiles(transform_cnn, n_partitions)

        for i, file_path in enumerate(files):
            # Move file to input folder as train train dataset
            file_path.rename(input_folder / "0")

            seeds.append(random.randint(0, 1000000))
            config.seed = seeds[-1]

            enc_model = train.main(config=config, output_name=f"{i}")
            local_models.append(enc_model)

        ### Aggregation section ###
        config = AggregationConfig(
            output_folder=output_final,
            input_folder=output_folder.parent,
            private_key=bytes(aggregation_key),
        )

        aggregate.main(config=config, output_name="model")

        ### Test final results ###
        final_model = load_model(output_final / "model")
        final_model.remove_noise_models(seeds)

        final_model.export_model(input_folder.parent / "algoCustomData.json")
        # with open(output_final / "model.pkl", "wb") as f:
        #     pickle.dump(final_model, f)

        model_test(final_model, "aggregated")

        # Test local models
        for i, (enc_model, seed) in enumerate(zip(local_models, seeds)):
            data = decrypt_nacl(bytes(aggregation_key), enc_model)
            model = load_model(data)
            model.remove_noise_models([seed])
            model_test(model, f"local_model_{i}")

    ### Fully trained model ###
    # Create custom data file (containing model definition)
    print("Full")
    with open(input_folder.parent / "algoCustomData.json", "w") as f:
        json.dump(model_def, f)
    new_model = load_model(input_folder.parent / "algoCustomData.json")
    for i in range(ITERATIONS):
        new_model.fit(x_train, y_train)
    model_test(new_model, "full_data")


if __name__ == "__main__":
    test_training(N_PARTITIONS)
