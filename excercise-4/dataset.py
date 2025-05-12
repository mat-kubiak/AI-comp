from torch.utils.data import DataLoader, random_split
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import pytorch_lightning as pl

class MnistDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers):
        super().__init__()
        self.train_loader = None
        self.val_dataset = None
        self.train_dataset = None
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        datasets.MNIST(root=self.data_dir, train=True, download=True)
        datasets.MNIST(root=self.data_dir, train=False, download=True)

    def setup(self, stage):
        entire_dataset = datasets.MNIST(root=self.data_dir, train=True,  transform=transforms.ToTensor(),  download=False)
        self.train_dataset, self.val_dataset = random_split(entire_dataset, [50000, 10000])
        self.train_loader = datasets.MNIST(root=self.data_dir, train=True,  transform=transforms.ToTensor(),  download=False)

    def train_dataloader(self):
        return DataLoader(self.train_loader, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)




class ImageNetteDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers):
        super().__init__()
        self.train_loader = None
        self.val_dataset = None
        self.train_dataset = None
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Define the transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),  # Resize to 160x160
            transforms.ToTensor(),
        ])

    def prepare_data(self):
        datasets.Imagenette(root=self.data_dir, size="160px", download=True)

    def setup(self, stage=None):
        # Apply the transformation to the dataset
        entire_dataset = datasets.Imagenette(root=self.data_dir, size="160px", transform=self.transform, download=False)

        # Randomly split the entire dataset
        self.train_dataset, self.val_dataset = random_split(
            entire_dataset, [8000, 1469]  # Adjust split if necessary
        )

    def train_dataloader(self):
        # Return the train DataLoader
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        # Return the validation DataLoader
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        # Return the test DataLoader (same as validation here)
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)