import torch
from torch.utils.data import DataLoader, random_split
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import pytorch_lightning as pl
import torch.nn.functional as Fnn

sobel_h = torch.tensor([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=torch.float32).view(1,1,3,3)

sobel_v = torch.tensor([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=torch.float32).view(1,1,3,3)

ext_methods = {
    'flatten': transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)) # flatten
    ]),
    'edges_sum': transforms.Compose([
        transforms.ToTensor(),  # (1, 28, 28)
        transforms.Lambda(lambda x: torch.stack([
            Fnn.conv2d(x.unsqueeze(0), sobel_h, padding=1).abs().sum(),
            Fnn.conv2d(x.unsqueeze(0), sobel_v, padding=1).abs().sum()
        ]))
    ]),
    'edges_mean': transforms.Compose([
        transforms.ToTensor(),  # (1, 28, 28)
        transforms.Lambda(lambda x: torch.stack([
            Fnn.conv2d(x.unsqueeze(0), sobel_h, padding=1).abs().mean(),
            Fnn.conv2d(x.unsqueeze(0), sobel_v, padding=1).abs().mean()
        ]))
    ]),
}

class MnistDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers, extraction_method='flatten'):
        super().__init__()
        self.train_loader = None
        self.val_dataset = None
        self.train_dataset = None
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.extraction_method = ext_methods[extraction_method]

    def prepare_data(self):
        datasets.MNIST(root=self.data_dir, train=True, download=True)
        datasets.MNIST(root=self.data_dir, train=False, download=True)

    def setup(self, stage):
        entire_dataset = datasets.MNIST(root=self.data_dir, train=True, transform=self.extraction_method, download=False)
        self.train_dataset, self.val_dataset = random_split(entire_dataset, [50000, 10000])
        self.train_loader = datasets.MNIST(root=self.data_dir, train=True, transform=self.extraction_method, download=False)

    def train_dataloader(self):
        return DataLoader(self.train_loader, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
