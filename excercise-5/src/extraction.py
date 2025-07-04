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
