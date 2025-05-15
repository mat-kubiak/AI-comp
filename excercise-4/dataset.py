import os
import random

from torch.utils.data import DataLoader, random_split, Subset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import pytorch_lightning as pl

import numpy as np

mnist_transforms = {
    'default': transforms.ToTensor(),
    'MK_random_rotate': transforms.Compose([
            transforms.RandomRotation(degrees=5),
            transforms.ToTensor()
        ]),
    'YourName_affine': transforms.Compose([
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=5,
        ),
        transforms.ToTensor()
    ])
}

imagenette_transforms = {
    'default': transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
        ]),
    'MK_random_crop': transforms.Compose([
            transforms.Resize((192, 192)),
            transforms.RandomCrop((160, 160)),
            transforms.ToTensor()
        ]),
    'JJ_color_jitter': transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor()
    ])
}


def limit_dataset_samples(dataset, limit, num_classes=10):
    class_indices = {i: [] for i in range(num_classes)}

    for idx, (data, target) in enumerate(dataset):
        class_indices[target].append(idx)

    sampled_indices = []
    samples_per_class = limit // num_classes

    for cls, indices in class_indices.items():
        sampled_indices += np.random.choice(indices, samples_per_class, replace=False).tolist()

    return Subset(dataset, sampled_indices)

def save_random_samples(dataset, save_dir, num_samples=3, prefix="sample"):
    os.makedirs(save_dir, exist_ok=True)

    indices = random.sample(range(len(dataset)), num_samples)

    for i, idx in enumerate(indices):
        img_tensor, label = dataset[idx]

        # Convert tensor to PIL image
        img = transforms.ToPILImage()(img_tensor)

        # Save with label and index for clarity
        img.save(os.path.join(save_dir, f"{prefix}_label{label}_{i}.png"))

class MnistDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers, transform='default', limit_samples=0):
        super().__init__()
        self.train_loader = None
        self.val_dataset = None
        self.train_dataset = None
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = mnist_transforms[transform]
        self.limit_samples = limit_samples

        self.num_classes = 10

        if self.limit_samples > 0 and self.limit_samples % self.num_classes != 0:
            raise ValueError(f"limit_samples ({self.limit_samples}) must be divisible by number of classes ({self.num_classes}).")

    def prepare_data(self):
        datasets.MNIST(root=self.data_dir, train=True, download=True)
        datasets.MNIST(root=self.data_dir, train=False, download=True)

    def setup(self, stage):
        self.train_dataset = datasets.MNIST(root=self.data_dir, train=True,  transform=self.transform,  download=False)
        self.val_dataset = datasets.MNIST(root=self.data_dir, train=False,  transform=transforms.ToTensor(),  download=False)

        if self.limit_samples > 0:
            self.train_dataset = limit_dataset_samples(self.train_dataset, self.limit_samples, self.num_classes)

        save_random_samples(self.train_dataset, "images_examples", 3)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


class ImageNetteDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers, transform='default', limit_samples=0):
        super().__init__()
        self.train_loader = None
        self.val_dataset = None
        self.train_dataset = None
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = imagenette_transforms[transform]
        self.limit_samples = limit_samples

        self.num_classes = 10

        if self.limit_samples > 0 and self.limit_samples % self.num_classes != 0:
            raise ValueError(f"limit_samples ({self.limit_samples}) must be divisible by number of classes ({self.num_classes}).")

    def prepare_data(self):
        datasets.Imagenette(root=self.data_dir, size="320px", download=True)


    def setup(self, stage=None):
        entire_dataset = datasets.Imagenette(root=self.data_dir, size="320px", transform=None, download=False)
        self.train_dataset, self.val_dataset = random_split(entire_dataset, [8000, 1469])

        self.train_dataset.dataset.transform = self.transform
        self.val_dataset.dataset.transform = imagenette_transforms['default']

        if self.limit_samples > 0:
            self.train_dataset = limit_dataset_samples(self.train_dataset, self.limit_samples, self.num_classes)

        save_random_samples(self.train_dataset, "images_examples", 3)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


