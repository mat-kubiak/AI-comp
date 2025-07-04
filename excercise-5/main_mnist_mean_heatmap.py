import torch
import numpy as np

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from captum.attr import IntegratedGradients

from src.mlp import NN
from src.cnn_mnist import CNN
from src.utils import plot_heatmap, collect_samples_per_class

def choose_dataset():
    print("Choose a mnist model:")
    print("1 - mlp")
    print("2 - cnn")
    choice = input("Enter number (1/2): ")

    if choice != '1' and choice != '2':
        choice = 1
        print("Invalid choice. Defaulting to MLP.")

    if choice == '1':
        model = NN.load_from_checkpoint('checkpoints/mnist_flatten_best_92.ckpt')
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1))  # Flatten the image into a vector
        ])
        model_type_str = 'MLP'
    elif choice == '2':
        model = CNN.load_from_checkpoint('checkpoints/mnist_cnn_best_99.ckpt')
        transform = transforms.ToTensor()
        model_type_str = 'CNN'

    dataset = datasets.MNIST(root='./datasets', train=False, transform=transform, download=True)
    return model, dataset, model_type_str

NUM_SAMPLES = 100

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, dataset, model_type_str = choose_dataset()
    model.to(device)
    model.eval()

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
        local_x = torch.tensor(X, dtype=torch.float32).to(device)
        outputs = model(local_x)
        y_pred = torch.argmax(outputs, dim=1)

    # Choose sample indices
    idx_per_class, samples_per_class, labels_per_class = collect_samples_per_class(X, y, y_pred, per_class=NUM_SAMPLES, only_correct=True)

    # Integrated Gradients
    print('\n### Integrated Gradents ###############\n')

    for cls in range(10):
        cls_samples = torch.tensor(np.array(samples_per_class[cls]), dtype=torch.float32).to(device)
        cls_labels = torch.tensor(np.array(labels_per_class[cls]), dtype=torch.int64).to(device)

        ig = IntegratedGradients(model)
        attributions, deltas = ig.attribute(cls_samples, target=cls_labels, return_convergence_delta=True)

        attributions = np.reshape(attributions.cpu(), (NUM_SAMPLES, 28, 28))
        mean_attrib = attributions.mean(dim=0)

        plot_heatmap(mean_attrib, f'Integrated Gradients ({model_type_str})\nMean of {NUM_SAMPLES} for class {cls}')

if __name__ == '__main__':
    main()
