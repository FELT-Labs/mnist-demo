"""Training function."""
import json
import random
import shutil
from pathlib import Path
from typing import Callable, List

from feltlabs.algorithm import aggregate, train
from feltlabs.config import AggregationConfig, TrainingConfig
from feltlabs.core.cryptography import decrypt_nacl
from feltlabs.core.models.base_model import BaseModel
from feltlabs.model import load_model
from nacl.public import PrivateKey

aggregation_key = PrivateKey.generate()


def _prepare_folders():
    """Prepare folders to use during training.

    Returns:
        folders: input_folder, output_folder, output_final_model
    """
    folder = Path("output")
    if folder.exists():
        shutil.rmtree(folder)

    # Create input, output folder structure as in Ocean
    folders = (
        folder / "input" / "fake_did",
        folder / "models" / "fake_did",
        folder / "final_model",
    )

    for f in folders:
        f.mkdir(parents=True, exist_ok=True)

    return folders


def _model_to_customdata(model_definition: dict, folder: Path) -> Path:
    """Create custom data file containing model definition."""
    custom_data_path = folder.parent / "algoCustomData.json"
    data = {"model_definition": model_definition}
    with open(custom_data_path, "w") as f:
        json.dump(data, f)
    return custom_data_path


def _decrypt_model(enc_model: bytes, seed: int) -> BaseModel:
    """Decrypt local model, used only for evaluation and testing."""
    data = decrypt_nacl(bytes(aggregation_key), enc_model)
    model = load_model(data)
    model.remove_noise_models([seed])
    return model


def federated_training(
    model_definition: dict,
    data_files: List[Path],
    eval_function: Callable,
    iterations: int = 1,
):
    """Train model based on definition and return evaluation for each iteration.

    Args:
        model_definition: dictionary describing model parameters
        data_files: list of data files each treated as separate dataset
        eval_function: function used for evaluation of models
        iterations: number of iterations of training to do
    """
    input_folder, output_folder, output_final = _prepare_folders()
    custom_data_path = _model_to_customdata(model_definition, input_folder)

    train_results = {f"local_model_{i}": [] for i in range(len(data_files))}
    train_results["aggregated"] = []

    for i in range(iterations):
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
        for i, file_path in enumerate(data_files):
            # Move file to input folder as train train dataset
            # During training on ocean files are named based on index starting from 0
            shutil.copy(file_path, input_folder / "0")

            config.seed = random.randint(0, 1000000)
            enc_model = train.main(config=config, output_name=f"{i}")
            seeds.append(config.seed)

            # Evaluate local model
            model = _decrypt_model(enc_model, config.seed)
            train_results[f"local_model_{i}"].append(eval_function(model))

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
        # Export model to use it in next iteration
        final_model.export_model(input_folder.parent / "algoCustomData.json")

        # Evaluate aggregated model
        train_results["aggregated"].append(eval_function(final_model))

    return final_model, train_results
