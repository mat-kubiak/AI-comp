import torch
import numpy as np

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from captum.robust import MinParamPerturbation

from src.mlp import NN
from src.cnn_mnist import CNN
from src.utils import plot_2_perturbed

def choose_dataset():
    print("Choose a mnist model:")
    print("1 - mlp")
    print("2 - cnn")
    choice = input("Enter number (1/2): ")

    if choice != '1' and choice != '2':
        choice = 1
        print("Invalid choice. Defaulting to flatten.")

    if choice == '1':
        model = NN.load_from_checkpoint('checkpoints/mnist_flatten_best_92.ckpt')
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1))  # Flatten the image into a vector
        ])
    elif choice == '2':
        model = CNN.load_from_checkpoint('checkpoints/mnist_cnn_best_99.ckpt')
        transform = transforms.ToTensor()

    dataset = datasets.MNIST(root='./datasets', train=False, transform=transform, download=True)
    return model, dataset

NUM_SAMPLES = 5

def gaussian_attack(input_tensor: torch.Tensor, std: float) -> torch.Tensor:
    noise = torch.randn_like(input_tensor) * std
    return input_tensor + noise

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, dataset = choose_dataset()
    model.to(device)
    model.eval()

    min_perturb = MinParamPerturbation(
        forward_func=model,
        attack=gaussian_attack,
        arg_name='std',
        arg_min=0.0,
        arg_max=2.0,
        arg_step=0.01
    )

    # Unpack data
    X, y = [], []
    for i in range(len(dataset)):
        xi, yi = dataset[i]  # this applies the transform!
        X.append(xi.numpy())  # assumes transform outputs a tensor
        y.append(yi)

    X = np.array(X)
    y = np.array(y)

    # Predict
    with torch.no_grad():
        t_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        outputs = model(t_tensor)
        preds = torch.argmax(outputs, dim=1)

    # remove pred misses
    mask = preds.cpu() == y
    X = X[mask]
    y = y[mask]

    # Choose sample indices
    random_indices = np.random.choice(len(X), size=NUM_SAMPLES, replace=False)
    samples = torch.tensor(X[random_indices], dtype=torch.float32).to(device)
    labels = torch.tensor(y[random_indices]).to(device)

    print("Random sample indices:", random_indices)
    print("Ground truth:", labels.cpu().numpy())

    for i in range(len(samples)):
        noised_image, min_std = min_perturb.evaluate(inputs=samples[i].unsqueeze(0), target=labels[i].unsqueeze(0))

        pred = torch.argmax(model(noised_image), dim=1).item()
        plot_2_perturbed(samples[i].cpu().reshape((28, 28)), noised_image.cpu().reshape((28,28)), f'Perturbed via Gaussian Noise (std={min_std:.04f})\nnew label: {pred}')

if __name__ == '__main__':
    main()
