import csv
import os


def setup_epoch_log(log_path):
    """Create the epoch-level log file with headers, if it doesn't already exist."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["run_id", "epoch", "train_loss", "val_loss"])


def log_epoch(log_path, run_id, epoch, train_loss, val_loss):
    """Append one epoch's summary to the epoch-level log."""
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([run_id, epoch, train_loss, val_loss])


def setup_batch_log(log_path):
    """Create the per-batch validation log file with headers, if it doesn't already exist."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["run_id", "epoch", "batch_idx", "val_loss"])


def log_batch(log_path, run_id, epoch, batch_idx, val_loss):
    """Append one batch's validation loss to the per-batch log."""
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([run_id, epoch, batch_idx, val_loss])