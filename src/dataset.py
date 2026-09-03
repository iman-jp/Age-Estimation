import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
import math
from collections import Counter


basic_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

class AgeDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = []
        self.ages = []
        for filename in os.listdir(image_dir):
            if filename.endswith(".jpg"):
                filepath = os.path.join(image_dir, filename)
                age = parse_age_from_filename(filename)
                if age is not None:
                    self.image_paths.append(filepath)
                    self.ages.append(age)
        self.transform = transform or basic_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, float(self.ages[idx])

def parse_age_from_filename(filename):
    filename = os.path.basename(filename)
    try:
        if filename.startswith("age"):
            age_str = filename[3:].split("_")[0]
        else:
            age_str = filename.split("_")[0]
        return int(age_str)
    except (ValueError, IndexError):
        return None

def compute_age_weights(dataset, cap_multiplier=10.0):
    age_counts = Counter(dataset.ages)
    inv_sqrt_freq = {age: 1.0 / math.sqrt(count) for age, count in age_counts.items()}
    avg = sum(inv_sqrt_freq.values()) / len(inv_sqrt_freq)
    weights = {age: w / avg for age, w in inv_sqrt_freq.items()}
    weights = {age: min(w, cap_multiplier) for age, w in weights.items()}
    return weights

def compute_age_weights_hybrid(dataset, bucket_threshold=85, bucket_size=10, cap_multiplier=10.0):
    """
    Per-exact-age weighting below bucket_threshold (where sample counts are
    large enough to trust individually), bucketed weighting above it (where
    individual ages have too few samples to weight reliably on their own).
    """
    age_counts = Counter(dataset.ages)

    age_to_bucket_key = {}
    effective_counts = {}
    for age, count in age_counts.items():
        if age < bucket_threshold:
            key = age
        else:
            key = bucket_threshold + ((age - bucket_threshold) // bucket_size) * bucket_size
        age_to_bucket_key[age] = key
        effective_counts[key] = effective_counts.get(key, 0) + count

    inv_sqrt_freq = {key: 1.0 / math.sqrt(c) for key, c in effective_counts.items()}
    avg = sum(inv_sqrt_freq.values()) / len(inv_sqrt_freq)
    bucket_weights = {key: min(w / avg, cap_multiplier) for key, w in inv_sqrt_freq.items()}

    age_weights = {age: bucket_weights[age_to_bucket_key[age]] for age in age_counts}
    return age_weights


if __name__ == "__main__":
    train_dataset = AgeDataset("/home/omid/Age-Estimation/data/train", transform=basic_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    dataset = AgeDataset("/home/omid/Age-Estimation/data/test", transform=basic_transform)
    print(f"Dataset size: {len(dataset)}")

    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    images, ages = next(iter(loader))
    print("Batch image shape:", images.shape)
    print("Batch ages shape:", ages.shape)
    print("First few ages:", ages[:5])

    # testing dataset distribution and weights
    # for split_name, split_dir in [
    #     ("train", "/home/omid/Age-Estimation/data/train"),
    #     ("val", "/home/omid/Age-Estimation/data/val"),
    #     ("test", "/home/omid/Age-Estimation/data/test"),
    # ]:
    #     dataset = AgeDataset(split_dir, transform=basic_transform)
    #     age_counts = Counter(dataset.ages)

    #     print(f"\n--- {split_name} ---")
    #     print(f"Total images: {len(dataset)}")
    #     print(f"Age range: {min(age_counts.keys())} to {max(age_counts.keys())}")
    #     print(f"Number of distinct ages present: {len(age_counts)}")

    #     sorted_counts = sorted(age_counts.items())
    #     rare_ages = [(age, count) for age, count in sorted_counts if count < 10]
    #     print(f"Ages with fewer than 10 images: {len(rare_ages)}")
    #     if rare_ages:
    #         print(f"  {rare_ages}")

    #     most_common = Counter(dataset.ages).most_common(5)
    #     print(f"5 most common ages: {most_common}")

    #     # write full per-age breakdown to CSV for later use in weight design
    #     import csv
    #     os.makedirs("logs", exist_ok=True)
    #     with open(f"logs/age_distribution_{split_name}.csv", "w", newline="") as f:
    #         writer = csv.writer(f)
    #         writer.writerow(["age", "count"])
    #         for age, count in sorted_counts:
    #             writer.writerow([age, count])
    #     print(f"Full distribution saved to logs/age_distribution_{split_name}.csv")