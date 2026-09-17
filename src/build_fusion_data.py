import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import csv
import torch
from torch.utils.data import DataLoader

from dataset import AgeDataset, basic_transform, load_blocklist
from model import build_age_model


models_path = {
    "pred_base": [
        "checkpoints/bs64_ep40_lr0.003_classWeightedHybrid_capMultiplier10.0_bucketThreshold65_bucketSize10_20260904_200247_softplus_with_blocklist.pt",
        "/home/omid/Age-Estimation/data/splits/val",
    ],
    "pred_upper_face": [
        "checkpoints/base_model_upper_face_masked_20260910_145207.pt",
        "/home/omid/Age-Estimation/data/masked/upper_face_masked/val",
    ],
    "pred_nose": [
        "checkpoints/base_model_nose_masked_20260910_140105.pt",
        "/home/omid/Age-Estimation/data/masked/nose_masked/val",
    ],
    "pred_lower_face": [
        "checkpoints/base_model_lower_face_masked_20260910_124234.pt",
        "/home/omid/Age-Estimation/data/masked/lower_face_masked/val",
    ],
    "pred_lips": [
        "checkpoints/base_model_lips_masked_20260910_105858.pt",
        "/home/omid/Age-Estimation/data/masked/lips_masked/val",
    ],
    "pred_eyes": [
        "checkpoints/base_model_eyes_masked_20260909_214026.pt",
        "/home/omid/Age-Estimation/data/masked/eyes_masked/val",
    ],
    "pred_eyes_extended": [
        "checkpoints/base_model_eyes_e_masked_20260910_085215.pt",
        "/home/omid/Age-Estimation/data/masked/eyes_e_masked/val",
    ],
    "pred_chin": [
        "checkpoints/base_model_chin_masked_20260910_094311.pt",
        "/home/omid/Age-Estimation/data/masked/chin_masked/val",
    ],
    "pred_bulls_eye": [
        "checkpoints/base_model_bulls_eye_masked_20260909_205044.pt",
        "/home/omid/Age-Estimation/data/masked/bulls_eye_masked/val",
    ],
}


def get_predictions(checkpoint_path, data_dir, blocklist, batch_size=32):
    model = build_age_model()
    model.model.to("cuda")

    checkpoint = torch.load(checkpoint_path)
    model.model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  Checkpoint: {checkpoint_path} (epoch {checkpoint['epoch']}, best val loss {checkpoint['best_val_loss']:.4f})")

    dataset = AgeDataset(data_dir, transform=basic_transform, blocked_filenames=blocklist.get("val", set()))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model.model.eval()
    predictions = {}
    true_ages = {}

    with torch.no_grad():
        for idx, (images, ages) in enumerate(loader):
            img_names = [os.path.basename(p) for p in dataset.image_paths[idx * batch_size: idx * batch_size + images.size(0)]]

            images = images.to("cuda")
            outputs = model.model(images)
            outputs = outputs[1]

            for i in range(images.size(0)):
                name = img_names[i]
                predictions[name] = outputs[i].item()
                true_ages[name] = ages[i].item()

    print(f"  {len(predictions)} predictions collected from {data_dir}")
    return predictions, true_ages


if __name__ == "__main__":
    blocklist = load_blocklist("/home/omid/Age-Estimation/logs/no_face_detection.csv")

    all_predictions = {}
    all_true_ages = {}

    for column_name, (checkpoint_path, data_dir) in models_path.items():
        print(f"\n--- {column_name} ---")
        preds, ages = get_predictions(checkpoint_path, data_dir, blocklist)
        all_predictions[column_name] = preds
        all_true_ages[column_name] = ages

    base_filenames = set(all_true_ages["pred_base"].keys())
    print(f"\nBase val set: {len(base_filenames)} images")

    for column_name, ages_dict in all_true_ages.items():
        cond_filenames = set(ages_dict.keys())
        missing = base_filenames - cond_filenames
        extra = cond_filenames - base_filenames
        if missing or extra:
            print(f"  WARNING: {column_name} misaligned — {len(missing)} missing, {len(extra)} unexpected extra filenames")
        else:
            print(f"  {column_name}: filenames match base exactly")

    common_filenames = base_filenames
    for ages_dict in all_true_ages.values():
        common_filenames &= set(ages_dict.keys())
    print(f"\nUsable images (present in all {len(models_path)} conditions): {len(common_filenames)}")

    output_path = "logs/fusion_training_data_from_val.csv"
    os.makedirs("logs", exist_ok=True)
    columns = list(models_path.keys())

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "true_age"] + columns)

        for filename in sorted(common_filenames):
            true_age = all_true_ages["pred_base"][filename]
            row = [filename, true_age]
            for col in columns:
                row.append(all_predictions[col][filename])
            writer.writerow(row)

    print(f"\nFusion training data written to: {output_path}")