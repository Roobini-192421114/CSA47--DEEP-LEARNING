from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGE_SIZE = 224

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=12),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return train_transform, eval_transform


def create_dataloaders(data_dir, batch_size=32, num_workers=0):
    data_dir = Path(data_dir)
    required = [data_dir / split for split in ("train", "val", "test")]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing split folder(s): " + ", ".join(missing) +
            ". Run prepare_dataset.py first."
        )

    train_tf, eval_tf = get_transforms()
    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=eval_tf)
    test_ds = datasets.ImageFolder(data_dir / "test", transform=eval_tf)

    if train_ds.classes != val_ds.classes or train_ds.classes != test_ds.classes:
        raise ValueError("Train, val and test folders must contain the same class subfolders.")

    if len(train_ds.classes) < 2:
        raise ValueError("At least two class folders are required.")

    loader_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": False
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_args)
    return train_loader, val_loader, test_loader, train_ds.classes
