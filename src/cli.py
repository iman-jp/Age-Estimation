import os
import csv
import sys
import joblib
from datetime import datetime

import click
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
from PIL import Image

from dataset import AgeDataset, basic_transform, load_blocklist, compute_age_weights_hybrid
from model import build_age_model
from train import train_one_epoch, train_one_epoch_weighted
from evaluate import validate
from logging_utils import setup_epoch_log, log_epoch

CLI_LOG_DIR = "CLILogs"
os.makedirs(CLI_LOG_DIR, exist_ok=True)

# create a cli container for training, inference, and evaluation commands
@click.group()
def cli():
    pass

# declare flags for the training command
@cli.command()
# each of these options corresponds to a parameter in the training function
@click.option("--train-dir", required=True)
@click.option("--val-dir", required=True)
@click.option("--batch-size", default=64)
@click.option("--epochs", default=30)
@click.option("--lr", default=0.001)
@click.option("--weighted/--unweighted", default=True)
@click.option("--bucket-threshold", default=65)
@click.option("--bucket-size", default=10)
@click.option("--cap-multiplier", default=10.0)
@click.option("--blocklist", default=None)
@click.option("--checkpoint-dir", default="checkpoints")
@click.option("--log-path", default=os.path.join(CLI_LOG_DIR, "training_log.csv"))
# run the existing training loop with the provided parameters
# example command:
# python3 src/cli.py train \
#   --train-dir data/train \
#   --val-dir data/val \
#   --batch-size 64 \
#   --epochs 2 \
#   --lr 0.001 \
#   --checkpoint-dir checkpoints \
#   --log-path logs/training_log.csv
def train(train_dir, val_dir, batch_size, epochs, lr, weighted, bucket_threshold,
          bucket_size, cap_multiplier, blocklist, checkpoint_dir, log_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_age_model()
    model.model.to(device)

    # blocklist is optional, mainly for us since we already filtered the mediapipe failures
    blocklist_map = load_blocklist(blocklist) if blocklist else {}

    train_dataset = AgeDataset(train_dir, transform=basic_transform, blocked_filenames=blocklist_map.get("train", set()))
    val_dataset = AgeDataset(val_dir, transform=basic_transform, blocked_filenames=blocklist_map.get("val", set()))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    loss_fn = nn.L1Loss()
    optimizer = optim.Adam(model.model.parameters(), lr=lr)

    run_id = f"cli_bs{batch_size}_ep{epochs}_lr{lr}_{'weighted' if weighted else 'plain'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_path = os.path.join(checkpoint_dir, run_id + ".pt")
    os.makedirs(checkpoint_dir, exist_ok=True)
    setup_epoch_log(log_path)

    age_weights = None
    if weighted:
        age_weights = compute_age_weights_hybrid(
            train_dataset,
            bucket_threshold=bucket_threshold,
            bucket_size=bucket_size,
            cap_multiplier=cap_multiplier,
        )

    best_val_loss = float("inf")

    for epoch in range(epochs):
        if weighted:
            train_loss = train_one_epoch_weighted(model, train_loader, optimizer, age_weights, default_weight=1.0, device=device)
        else:
            train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device=device)

        val_loss = validate(model, val_loader, loss_fn, device=device)
        print(f"Epoch {epoch+1}/{epochs} — train loss: {train_loss:.4f} — val loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.model.state_dict(),
                "best_val_loss": best_val_loss,
                "epoch": epoch + 1,
            }, checkpoint_path)
            print(f"Saved new best model at epoch {epoch+1} with val loss: {best_val_loss:.4f}")

        log_epoch(log_path, run_id, epoch + 1, train_loss, val_loss)

    print(f"\ndone. best checkpoint: {checkpoint_path} (val loss {best_val_loss:.4f})")


@cli.command()
@click.option("--checkpoint", required=True)
@click.option("--input-dir", required=True)
@click.option("--output-csv", default=os.path.join(CLI_LOG_DIR, "inference_results.csv"))
# batch inference / load a trining checkpoint then loops over all images in the input directory and writes predictions to a CSV
# example command:
# python3 src/cli.py infer \
#   --checkpoint checkpoints/base_model.pt \
#   --input-dir data/splits/test \
#   --output-csv logs/infer_test_output.csv
def infer(checkpoint, input_dir, output_csv):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_age_model()
    model.model.to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.model.load_state_dict(ckpt["model_state_dict"])
    model.model.eval()

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "predicted_age"])

        with torch.no_grad():
            for filename in image_files:
                image = Image.open(os.path.join(input_dir, filename)).convert("RGB")
                image_tensor = basic_transform(image).unsqueeze(0).to(device)

                output = model.model(image_tensor)
                # eval mode still returns that softmax tuple thing from the old classify head
                predicted_age = output[1] if isinstance(output, tuple) else output

                writer.writerow([filename, round(predicted_age.item(), 2)])
                print(f"{filename}: {predicted_age.item():.2f}")

    print(f"\nresults written to {output_csv}")


@cli.command(name="evaluate")
@click.option("--checkpoint", required=True)
@click.option("--test-dir", required=True)
@click.option("--blocklist", default=None)
@click.option("--output-csv", default=os.path.join(CLI_LOG_DIR, "test_results.csv"))
# run a full evaluation on a test set, logging predictions and the overall MAE to a CSV
# example command:
# python3 src/cli.py evaluate \
#   --checkpoint checkpoints/base_model.pt \
#   --test-dir /home/omid/Age-Estimation/data/splits/test \
#   --output-csv logs/base_model_test_results.csv
def evaluate_cmd(checkpoint, test_dir, blocklist, output_csv):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_age_model()
    model.model.to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.model.load_state_dict(ckpt["model_state_dict"])
    print(f"Checkpoint: {checkpoint}")
    print(f"  (trained to epoch {ckpt['epoch']}, best val loss during training: {ckpt['best_val_loss']:.4f})")

    blocklist_map = load_blocklist(blocklist) if blocklist else {}
    test_dataset = AgeDataset(test_dir, transform=basic_transform, blocked_filenames=blocklist_map.get("test", set()))
    if len(test_dataset) == 0:
        print(f"No images found in {test_dir} — check the path is correct.")
        return
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    model.model.eval()
    total_abs_error = 0.0
    num_samples = 0
    rows = []

    with torch.no_grad():
        for idx, (images, ages) in enumerate(test_loader):
            names = [os.path.basename(p) for p in test_dataset.image_paths[idx * 32: idx * 32 + images.size(0)]]
            images = images.to(device)
            ages_gpu = ages.to(device).float().unsqueeze(1)

            output = model.model(images)
            output = output[1] if isinstance(output, tuple) else output

            abs_error = torch.abs(output - ages_gpu)
            total_abs_error += abs_error.sum().item()
            num_samples += images.size(0)

            for i in range(images.size(0)):
                rows.append([names[i], ages[i].item(), output[i].item()])

    test_mae = total_abs_error / num_samples
    print(f"Test set MAE ({num_samples} images): {test_mae:.4f} years")

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "actual_age", "predicted_age"])
        writer.writerows(rows)
        writer.writerow([])
        writer.writerow(["TEST_MAE", test_mae])
    print(f"logged detailed results to: {output_csv}")

# use this dictionary to map model names to their corresponding checkpoint paths for the model-test command
MODEL_SHORTCUTS = {
    "base": "checkpoints/base_model.pt",
    "bulls_eye_masked": "checkpoints/base_model_bulls_eye_masked_20260909_205044.pt",
    "chin_masked": "checkpoints/base_model_chin_masked_20260910_094311.pt",
    "eyes_extended_masked": "checkpoints/base_model_eyes_e_masked_20260910_085215.pt",
    "eyes_masked": "checkpoints/base_model_eyes_masked_20260909_214026.pt",
    "lips_masked": "checkpoints/base_model_lips_masked_20260910_105858.pt",
    "lower_face_masked": "checkpoints/base_model_lower_face_masked_20260910_124234.pt",
    "nose_masked": "checkpoints/base_model_nose_masked_20260910_140105.pt",
    "upper_face_masked": "checkpoints/base_model_upper_face_masked_20260910_145207.pt",
}

# test a single image with a specified model checkpoint, printing the predicted age
# example command:
# python3 src/cli.py model-test \
#   --model base \
#   --image data/splits/test/00001.jpg 
@cli.command(name="model-test")
@click.option("--model", type=click.Choice(list(MODEL_SHORTCUTS.keys())), required=True)
@click.option("--image", required=True)
def model_test(model, image):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_path = MODEL_SHORTCUTS[model]
    model_obj = build_age_model()
    model_obj.model.to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_obj.model.load_state_dict(ckpt["model_state_dict"])
    model_obj.model.eval()

    img = Image.open(image).convert("RGB")
    image_tensor = basic_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model_obj.model(image_tensor)
        predicted_age = output[1] if isinstance(output, tuple) else output

    print(f"{model}: predicted age {predicted_age.item():.2f}")

# test a single image with running it throw the mediapipe face landmark detection, 
# applying the various masking variants, and then running the fusion model to get a final predicted age
# example command:
# python3 src/cli.py model-fusion \
#   --image data/splits/test/age39_069711.jpg \
#   --method bayesian
@cli.command(name="model-fusion")
@click.option("--image", required=True)
@click.option("--method", type=click.Choice(["tree", "forest", "bayesian"]), default="bayesian")
def model_fusion(image, method):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "masking"))
    from mediapipe_init import media_pipe
    from masking import detect_face_landmarks, apply_masking_variant, VARIANT_DEFINITIONS

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # variant name -> (checkpoint path, feature column name)
    conditions = {
        "bulls_eye_masked":  ("checkpoints/base_model_bulls_eye_masked_20260909_205044.pt", "pred_bulls_eye"),
        "chin_masked":       ("checkpoints/base_model_chin_masked_20260910_094311.pt", "pred_chin"),
        "eyes_e_masked":      ("checkpoints/base_model_eyes_e_masked_20260910_085215.pt", "pred_eyes_extended"),
        "eyes_masked":        ("checkpoints/base_model_eyes_masked_20260909_214026.pt", "pred_eyes"),
        "lips_masked":        ("checkpoints/base_model_lips_masked_20260910_105858.pt", "pred_lips"),
        "lower_face_masked":  ("checkpoints/base_model_lower_face_masked_20260910_124234.pt", "pred_lower_face"),
        "nose_masked":        ("checkpoints/base_model_nose_masked_20260910_140105.pt", "pred_nose"),
        "upper_face_masked":  ("checkpoints/base_model_upper_face_masked_20260910_145207.pt", "pred_upper_face"),
    }

    print("detecting face landmarks...")
    face_landmarker, pose_landmarker = media_pipe()
    status, image_np, landmarks, w, h = detect_face_landmarks(image, face_landmarker, pose_landmarker)

    if status == "failed":
        print("no face detected in this image, cannot run fusion")
        return

    def predict_from_array(checkpoint_path, img_array):
        m = build_age_model()
        m.model.to(device)
        ckpt = torch.load(checkpoint_path, map_location=device)
        m.model.load_state_dict(ckpt["model_state_dict"])
        m.model.eval()
        pil_img = Image.fromarray(img_array).convert("RGB")
        tensor = basic_transform(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = m.model(tensor)
            out = out[1] if isinstance(out, tuple) else out
        return out.item()

    predictions = {"pred_base": predict_from_array("checkpoints/base_model.pt", image_np)}
    print(f"pred_base: {predictions['pred_base']:.2f}")

    for variant_name, (checkpoint_path, feature_name) in conditions.items():
        indices_groups = VARIANT_DEFINITIONS[variant_name]
        masked_np = apply_masking_variant(image_np, landmarks, w, h, indices_groups)
        predictions[feature_name] = predict_from_array(checkpoint_path, masked_np)
        print(f"{feature_name}: {predictions[feature_name]:.2f}")

    fusion_model = joblib.load(f"checkpoints/fusion_model_{method}.joblib")
    feature_order = ["pred_base", "pred_upper_face", "pred_nose", "pred_lower_face",
                      "pred_lips", "pred_eyes", "pred_eyes_extended", "pred_chin", "pred_bulls_eye"]
    X = pd.DataFrame([[predictions[col] for col in feature_order]], columns=feature_order)

    face_landmarker.close()
    pose_landmarker.close()

    final_prediction = fusion_model.predict(X)[0]
    print(f"\nfusion ({method}) predicted age: {final_prediction:.2f}")

if __name__ == "__main__":
    cli()