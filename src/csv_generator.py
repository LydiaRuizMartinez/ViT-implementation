import os
import csv
import torch
from src.data import load_data
from src.train_functions import t_step
from src.utils import set_seed

def main() -> None:
    # set seed and device.
    set_seed(42)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # load test data
    _, _, test_data = load_data("data", batch_size=16, num_workers=4)
    
    # define directories
    models_dir = os.path.join(os.getcwd(), "models")
    csv_dir = os.path.join(os.getcwd(), "csv")
    os.makedirs(csv_dir, exist_ok=True)
    csv_file = os.path.join(csv_dir, "experiment_summary.csv")
    
    # CSV header fields.
    fieldnames = ["MODELO", "LR", "BATCH SIZE", "EPOCHS", "ACCURACY"]
    
    # ppen CSV file in write mode
    with open(csv_file, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for filename in os.listdir(models_dir):
            if filename.endswith(".pt"):
                base_name = filename[:-3]
                parts = base_name.split("_")
                if len(parts) < 5:
                    print(f"Filename {filename} does not match expected format. Skipping.")
                    continue
                lr_value = parts[2]
                batch_size_value = parts[3]
                epochs_value = parts[4]
                
                model_path = os.path.join(models_dir, filename)
                try:
                    model = torch.jit.load(model_path).to(device)
                except Exception as e:
                    print(f"Error loading model {filename}: {e}")
                    continue
                
                accuracy: float = t_step(model, test_data, device)
                print(f"Processed {filename}: Accuracy = {accuracy:.4f}")
                
                writer.writerow({
                    "MODELO": base_name,
                    "LR": lr_value,
                    "BATCH SIZE": batch_size_value,
                    "EPOCHS": epochs_value,
                    "ACCURACY": accuracy
                })
                
    print(f"CSV generated at {csv_file}")

if __name__ == "__main__":
    main()
