import torch
from torch.utils.data import DataLoader, random_split
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import pytorch_lightning as pl
import torch.nn.functional as Fnn



def quadrant_white_pixel_count(x):
    x = x.squeeze(0)  # shape: (28, 28)

    # Split the image into 4 quadrants
    top_left = x[:14, :14]
    top_right = x[:14, 14:]
    bottom_left = x[14:, :14]
    bottom_right = x[14:, 14:]

    # Count white pixels in each quadrant
    top_left_count = (top_left > 0.5).sum().item()
    top_right_count = (top_right > 0.5).sum().item()
    bottom_left_count = (bottom_left > 0.5).sum().item()
    bottom_right_count = (bottom_right > 0.5).sum().item()

    return torch.tensor([top_left_count, top_right_count, bottom_left_count, bottom_right_count], dtype=torch.float32)


def white_black_ratio_in_halves(x):
    x = x.squeeze(0)  # Ensure the shape is (28, 28)

    # Split the image into top and bottom halves (14 rows each)
    top_half = x[:14, :]  # First 14 rows (top half)
    bottom_half = x[14:, :]  # Last 14 rows (bottom half)

    # Convert both halves to binary (white = 1, black = 0)
    binary_top_half = (top_half > 0.5).float()
    binary_bottom_half = (bottom_half > 0.5).float()

    # Count white and black pixels in each half
    white_top = binary_top_half.sum().item()
    black_top = binary_top_half.numel() - white_top

    white_bottom = binary_bottom_half.sum().item()
    black_bottom = binary_bottom_half.numel() - white_bottom

    # Calculate white to black ratios for each half
    # Prevent division by zero if there are no black pixels (add small epsilon)
    epsilon = 1e-6
    ratio_top = white_top / (black_top + epsilon)
    ratio_bottom = white_bottom / (black_bottom + epsilon)

    return torch.tensor([ratio_top, ratio_bottom], dtype=torch.float32)

sobel_h = torch.tensor([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=torch.float32).view(1,1,3,3)

sobel_v = torch.tensor([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=torch.float32).view(1,1,3,3)

sobel_d1 = torch.tensor([[0, 1, 2],
                        [-1, 0, 1],
                        [-2, -1, 0]], dtype=torch.float32).view(1,1,3,3)

sobel_d2 = torch.tensor([[2, 1, 0],
                        [1, 0, -1],
                        [0, -1, -2]], dtype=torch.float32).view(1,1,3,3)

ext_methods = {
    'flatten': transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten the image into a vector
    ]),
    'edges_sum': transforms.Compose([
        transforms.ToTensor(),  # (1, 28, 28)
        transforms.Lambda(lambda x: torch.stack([
            Fnn.conv2d(x.unsqueeze(0), sobel_h, padding=1).abs().sum(),
            Fnn.conv2d(x.unsqueeze(0), sobel_v, padding=1).abs().sum()
        ]))
    ]),
    'edges_all': transforms.Compose([
        transforms.ToTensor(),  # (1, 28, 28)
        transforms.Lambda(lambda x: torch.stack([
            Fnn.conv2d(x.unsqueeze(0), sobel_h, padding=1).abs().sum(),
            Fnn.conv2d(x.unsqueeze(0), sobel_v, padding=1).abs().sum(),
            Fnn.conv2d(x.unsqueeze(0), sobel_d1, padding=1).abs().sum(),
            Fnn.conv2d(x.unsqueeze(0), sobel_d2, padding=1).abs().sum(),
            x.mean()
        ]))
    ]),
    'wbrh': transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: white_black_ratio_in_halves(x))
    ]),
    'qwpc': transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: quadrant_white_pixel_count(x))
    ])
}

class MnistDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers, extraction_method):
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


from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl

class SklearnDataModule(pl.LightningDataModule):
    def __init__(self, dataset_name, batch_size, num_workers, test_size=0.2):
        super().__init__()
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.test_size = test_size
        self.num_workers = num_workers

    def prepare_data(self):
        if self.dataset_name == "Iris":
            data = load_iris()
            print('Loaded iris')
        elif self.dataset_name == "Wine":
            print('Loaded wine')
            data = load_wine()
        elif self.dataset_name == "Breast Cancer Wisconsin":
            print('Loaded wisconsin')
            data = load_breast_cancer()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        X = data['data']
        y = data['target']

        self.input_size = X.shape[1]
        self.num_classes = len(set(y))

        # Normalize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_scaled, y, test_size=self.test_size, random_state=42, stratify=y
        )

    def setup(self, stage=None):
        self.train_dataset = TensorDataset(
            torch.tensor(self.X_train, dtype=torch.float32),
            torch.tensor(self.y_train, dtype=torch.long)
        )
        self.val_dataset = TensorDataset(
            torch.tensor(self.X_val, dtype=torch.float32),
            torch.tensor(self.y_val, dtype=torch.long)
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
