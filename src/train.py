# deep learning libraries
import torch
from torch.utils.tensorboard import SummaryWriter
from src.train_functions import train_step, val_step

# other libraries
from tqdm.auto import tqdm
import csv
import os

# own modules
from src.models import ViTForClassification  
from src.data import load_data
from src.utils import (
    save_model,
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
NUMBER_OF_CLASSES: int = 10

# configuration of de ViT
config = {
    "image_size": 64,           
    "patch_size": 8,            
    "num_channels": 3,
    "hidden_size": 256,          
    "num_hidden_layers": 8,      
    "num_attention_heads": 8,   
    "intermediate_size": 512,    
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    "num_classes": NUMBER_OF_CLASSES,
    "qkv_bias": True,
    "initializer_range": 0.02,
}

def main() -> None:
    """
    This function is the main program for the training.
    """

    epochs: int = 30
    lr: float = 5e-4
    batch_size: int = 128

    # clear previous log file
    open("nohup.out", "w").close()

    # load training and validation data
    train_data, val_data, _ = load_data(DATA_PATH, batch_size=batch_size, num_workers=4)

    # define model name for logging
    name: str = f"model_lr_{lr}_{batch_size}_{epochs}"
    writer: SummaryWriter = SummaryWriter(f"runs/{name}")

    # initialize model and move to device
    model: torch.nn.Module = ViTForClassification(config).to(device)

    # define loss function, optimizer, and learning rate scheduler
    loss: torch.nn.Module = torch.nn.CrossEntropyLoss()
    optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    try:
        for epoch in tqdm(range(epochs), desc="epochs", position=0):
            # perform training and validation steps
            train_step(model, train_data, loss, optimizer, writer, epoch, device)
            val_step(model, val_data, loss, writer, epoch, device)
            
            # update learning rate
            scheduler.step()

    except KeyboardInterrupt:
        save_model(model, name)

    # final model save
    save_model(model, name)

if __name__ == "__main__":
    main()
