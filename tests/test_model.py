import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import csv
import torch
from torch.utils.data import DataLoader

from dataset import AgeDataset, basic_transform
from model import build_age_model


LOGGING_ENABLED = True  


def evaluate_on_test(checkpoint_path, test_dir, batch_size=1, log_path=None):
    model = build_age_model()
    model.model.to("cuda")

    checkpoint = torch.load(checkpoint_path)
    model.model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Checkpoint: {checkpoint_path}")
    print(f"  (trained to epoch {checkpoint['epoch']}, best val loss during training: {checkpoint['best_val_loss']:.4f})")

    test_dataset = AgeDataset(test_dir, transform=basic_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model.model.eval()
    total_abs_error = 0.0
    num_samples = 0
    results_rows = []

    with torch.no_grad():
        for idx, (images, ages) in enumerate(test_loader):
            img_names = [os.path.basename(p) for p in test_dataset.image_paths[idx * batch_size : idx * batch_size + images.size(0)]]

            images = images.to("cuda")
            ages_gpu = ages.to("cuda").float().unsqueeze(1)

            outputs = model.model(images)
            outputs = outputs[1]

            abs_error = torch.abs(outputs - ages_gpu)
            total_abs_error += abs_error.sum().item()
            num_samples += images.size(0)

            for i in range(images.size(0)):
                name = img_names[i]
                actual = ages[i].item()
                predicted = outputs[i].item()
                results_rows.append([name, actual, predicted])

    test_mae = total_abs_error / num_samples
    print(f"Test set MAE ({num_samples} images): {test_mae:.4f} years")

    if LOGGING_ENABLED and log_path is not None:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "actual_age", "predicted_age"])
            writer.writerows(results_rows)
            writer.writerow([])
            writer.writerow(["TEST_MAE", test_mae])
        print(f"Logged detailed results to: {log_path}")

    return test_mae


if __name__ == "__main__":
    checkpoint_path = "checkpoints/bs64_ep40_lr0.003_classWeightedHybrid_capMultiplier10.0_bucketThreshold65_bucketSize10_20260830_110906.pt" 
    test_dir = "/home/omid/Age-Estimation/data/test"
    log_path = "logs/test_results.csv"

    evaluate_on_test(checkpoint_path, test_dir, log_path=log_path)