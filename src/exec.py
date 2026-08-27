import os
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from dataset import AgeDataset, basic_transform
from model import build_age_model
from train import train_one_epoch
from evaluate import validate
from logging_utils import setup_epoch_log, log_epoch, setup_batch_log
from datetime import datetime


if __name__ == "__main__":

    hyperparameters = {
        "batch_size": 128,
        "start_epoch": 0,
        "num_epochs": 50,
        "learning_rate": 0.001,
        "loss_function": nn.L1Loss(),
    }
    
    # an unique identifier for each run 
    hyperparameters["run_id"] = (
        f"bs{hyperparameters['batch_size']}"
        f"_ep{hyperparameters['num_epochs']}"
        f"_lr{hyperparameters['learning_rate']}"
        f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    model = build_age_model()
    model.model.to("cuda")

    run_id = hyperparameters["run_id"]
    checkpoint_path = f"checkpoints/{run_id}.pt"
    os.makedirs("checkpoints", exist_ok=True)

    best_val_loss = float("inf")
    start_epoch = hyperparameters["start_epoch"]

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.model.load_state_dict(checkpoint["model_state_dict"])
        best_val_loss = checkpoint["best_val_loss"]
        start_epoch = checkpoint["epoch"]
        print(f"Resumed from checkpoint: epoch {start_epoch}, best val loss so far: {best_val_loss:.4f}")
    else:
        print("No checkpoint found, starting fresh")

    train_dataset = AgeDataset("/home/omid/Age-Estimation/data/train", transform=basic_transform)
    val_dataset = AgeDataset("/home/omid/Age-Estimation/data/val", transform=basic_transform)

    train_loader = DataLoader(train_dataset, batch_size=hyperparameters["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=hyperparameters["batch_size"], shuffle=False, num_workers=4, pin_memory=True)

    loss_fn = hyperparameters["loss_function"]
    optimizer = optim.Adam(model.model.parameters(), lr=hyperparameters["learning_rate"])

    num_epochs = hyperparameters["num_epochs"]

    log_path = "logs/training_log.csv"
    batch_log_path = "logs/val_batch_log.csv"
    setup_epoch_log(log_path)
    setup_batch_log(batch_log_path)

    for epoch in range(start_epoch, num_epochs):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer)
        val_loss = validate(model, val_loader, loss_fn, run_id=run_id, epoch=epoch + 1, batch_log_path=batch_log_path)
        print(f"Epoch {epoch+1}/{num_epochs} — train loss: {train_loss:.4f} — val loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.model.state_dict(),
                "best_val_loss": best_val_loss,
                "epoch": epoch + 1,
            }, checkpoint_path)
            print(f"Saved new best model at epoch {epoch+1} with val loss: {best_val_loss:.4f}")

        log_epoch(log_path, run_id, epoch + 1, train_loss, val_loss)
