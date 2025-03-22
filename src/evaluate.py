# deep learning libraries
import torch
from torch.jit import RecursiveScriptModule
import os
import csv

# own modules
from src.data import load_data
from src.train_functions import t_step
from src.utils import (
    set_seed,
)

# set device
device: torch.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

# set all seeds and set number of threads
set_seed(42)
torch.set_num_threads(8)

# static variables
DATA_PATH: str = "data"


def main(name: str) -> float:
    """
    This function is the main program for the testing.
    """

    # load dataset
    (
        train_data,
        val_data,
        test_data,
    ) = load_data(DATA_PATH, batch_size=16, num_workers=4)

    # load pre-trained model
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, "..", "models", f"{name}.pt")

    model: RecursiveScriptModule = torch.jit.load(model_path).to(device)

    # evaluate model performance on the test set
    accuracy: float = t_step(model, test_data, device)

    return accuracy


if __name__ == "__main__":
    print(f"accuracy: {main('best_model')}")
