from torch.utils.data import DataLoader, random_split
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import pytorch_lightning as pl

mnist_transforms = {
    'default': transforms.ToTensor(),
    'MK_random_rotate': transforms.Compose([
            transforms.RandomRotation(degrees=5),
            transforms.ToTensor()
        ])
}

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

    def prepare_data(self):
        datasets.MNIST(root=self.data_dir, train=True, download=True)
        datasets.MNIST(root=self.data_dir, train=False, download=True)

    def setup(self, stage):
        entire_dataset = datasets.MNIST(root=self.data_dir, train=True,  transform=self.transform,  download=False)
        self.train_dataset, self.val_dataset = random_split(entire_dataset, [50000, 10000])
        self.train_loader = datasets.MNIST(root=self.data_dir, train=True,  transform=self.transform,  download=False)

    def train_dataloader(self):
        return DataLoader(self.train_loader, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

imagenette_transforms = {
    'default': transforms.Compose([
            transforms.Resize((160, 160)),  # Resize to 160x160
            transforms.ToTensor(),
        ]),
    'MK_random_crop': transforms.Compose([
            transforms.Resize((192, 192)),
            transforms.RandomCrop((160, 160)),
            transforms.ToTensor()
        ])
}

class ImageNetteDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers, transform='default'):
        super().__init__()
        self.train_loader = None
        self.val_dataset = None
        self.train_dataset = None
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = imagenette_transforms[transform]

    def prepare_data(self):
        datasets.Imagenette(root=self.data_dir, size="160px", download=True)

    def setup(self, stage=None):
        entire_dataset = datasets.Imagenette(root=self.data_dir, size="160px", transform=self.transform, download=False)
        self.train_dataset, self.val_dataset = random_split(entire_dataset, [8000, 1469])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
