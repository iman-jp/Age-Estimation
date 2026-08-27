import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader


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