"""
Data loading for the Oxford-IIIT Pet dataset (37 breeds).

Builds the preprocessing pipelines, the stratified train/validation split
of the official trainval partition, and the DataLoaders used by all three
transfer-learning approaches.
"""

import torch
import csv
from collections import Counter
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import MobileNet_V3_Small_Weights
from sklearn.model_selection import train_test_split

DATA_ROOT = Path("data")      # where torchvision downloads the dataset
SPLITS_DIR = Path("splits")
RESULTS_DIR = Path("results")
VAL_FRACTION = 0.15           # fraction of trainval used for validation
SEED = 42                     # seed for the train/val split

def save_split(path, indices, dataset):
    """Write one row per image: index, file name, label, breed name."""

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "filename", "label", "breed"])
        for i in indices:
            label = dataset._labels[i]
            writer.writerow([i, Path(dataset._images[i]).name, label, dataset.classes[label]])

def load_split(path):
    """Read back the list of indices from a split file."""
    with open(path, newline="") as f:
        return [int(row["index"]) for row in csv.DictReader(f)]


def get_dataloaders(batch_size=32):
    """Build the train/val/test DataLoaders (stratified split, saved in splits/)."""

    SPLITS_DIR.mkdir(exist_ok=True)    
    RESULTS_DIR.mkdir(exist_ok=True)

    # Val/test: official preprocessing of the pre-trained weights (resize 256, center crop 224, ImageNet norm)
    weights = MobileNet_V3_Small_Weights.DEFAULT
    eval_transform = weights.transforms()


    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),   # random crop, output size expected by the network
        transforms.RandomHorizontalFlip(),   # left/right mirroring
        transforms.ToTensor(),
        transforms.Normalize(eval_transform.mean, eval_transform.std),  # same ImageNet stats as the weights
    ])


    # Same trainval images, two transforms: train subset gets augmentation, val subset does not
    trainval_train = datasets.OxfordIIITPet(root=DATA_ROOT, split="trainval", target_types="category", download=True, transform=train_transform)
    trainval_val  = datasets.OxfordIIITPet(root=DATA_ROOT, split="trainval", target_types="category", download=True, transform=eval_transform)
    test_dataset   = datasets.OxfordIIITPet(root=DATA_ROOT, split="test", target_types="category", download=True, transform=eval_transform)


    train_file = SPLITS_DIR / "train.csv"
    val_file = SPLITS_DIR / "val.csv"

    if train_file.exists() and val_file.exists():
        # Split already created: reuse it
        train_idx = load_split(train_file)
        val_idx = load_split(val_file)
    else:
        # First run: create the stratified split and save it
        indices = list(range(len(trainval_val)))
        train_idx, val_idx = train_test_split(
            indices,
            test_size=VAL_FRACTION,
            stratify=trainval_val._labels,
            random_state=SEED,
        )
        save_split(train_file, train_idx, trainval_val)
        save_split(val_file, val_idx, trainval_val)


    train_dataset = Subset(trainval_train, train_idx)
    val_dataset = Subset(trainval_val, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    # Images per breed in each split
    classes = trainval_val.classes

    train_labels = [trainval_val._labels[i] for i in train_idx]
    val_labels   = [trainval_val._labels[i] for i in val_idx]
    test_labels  = test_dataset._labels

    train_counts = Counter(train_labels)
    val_counts   = Counter(val_labels)
    test_counts  = Counter(test_labels)

    with open(RESULTS_DIR / "class_counts.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["breed", "train", "val", "test"])
        for c, name in enumerate(classes):
            writer.writerow([name, train_counts[c], val_counts[c], test_counts[c]])
        writer.writerow(["Total", len(train_labels), len(val_labels), len(test_labels)])

    print(f"Split: {len(train_idx)} train, {len(val_idx)} val, {len(test_labels)} test")

    return train_loader, val_loader, test_loader, classes

if __name__ == "__main__":
    train_loader, val_loader, test_loader, classes = get_dataloaders()