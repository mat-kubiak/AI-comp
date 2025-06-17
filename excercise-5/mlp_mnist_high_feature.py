import torch
import numpy as np
import matplotlib.pyplot as plt

from mlp import NN
from cnn_mnist import CNN

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from captum.attr import IntegratedGradients
from captum.attr import Lime
from captum._utils.models.linear_model import SkLearnLinearRegression, SkLearnLasso

from skimage.segmentation import slic

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

def shuffle_by_value(data):
    """
        shuffles unique values of an array
        (i.e. all 3s become 6s and all 6s become 3s)
    """
    unique_data = np.unique(data)
    shuffled_labels = np.random.permutation(unique_data)

    label_map = dict(zip(unique_data, shuffled_labels))

    return np.vectorize(label_map.get)(data)

def plot_2_part_heatmap(original_data, attr_data, title=''):
    """
    plot original image and a heatmap
    """
    max_val = np.abs(attr_data).max()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # Original image
    axes[0].imshow(original_data, cmap='gray')
    axes[0].set_title('Original Image')

    # Attribution heatmap
    im = axes[1].imshow(attr_data, vmin=-max_val, vmax=max_val, cmap='bwr')
    axes[1].set_title('Attributions')
    fig.colorbar(im, ax=axes[1])

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    plt.show()

def plot_3_part_heatmap(original_data, segment_data, attr_data, title=''):
    """
    plot original image, SLIC segmentation and a heatmap
    """
    max_val = np.abs(attr_data).max()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Original image
    axes[0].imshow(original_data, cmap='gray')
    axes[0].set_title('Original Image')

    # Segmentation map
    shuffled_segments = shuffle_by_value(segment_data)
    axes[1].imshow(shuffled_segments, cmap='nipy_spectral')
    axes[1].set_title('Segmentation')

    # Attribution heatmap
    im = axes[2].imshow(attr_data, vmin=-max_val, vmax=max_val, cmap='bwr')
    axes[2].set_title('Attributions')
    fig.colorbar(im, ax=axes[2])

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    plt.show()

NUM_SAMPLES = 5

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, dataset = choose_dataset()
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

    # Choose sample indices
    random_indices = np.random.choice(len(X), size=NUM_SAMPLES, replace=False)
    samples = torch.tensor(X[random_indices], dtype=torch.float32).to(device)
    labels = torch.tensor(y[random_indices]).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(samples)
        preds = torch.argmax(outputs, dim=1)
    print("Random sample indices:", random_indices)
    print("Predictions: ", preds.cpu().numpy())
    print("Ground truth:", labels.cpu().numpy())

    # Captum Integrated Gradients
    print('\n### Integrated Gradents ###############\n')
    ig = IntegratedGradients(model)
    attributions, delta = ig.attribute(samples, target=labels, return_convergence_delta=True)

    attributions = np.reshape(attributions.cpu(), (5, 28, 28))
    for i in range(len(attributions)):
        plot_2_part_heatmap(np.reshape(samples[i].cpu(), (28, 28)), attributions[i], f'Integrated Gradients\ny_true: {labels[i]}, y_pred: {preds[i]}')

    # LIME
    print('\n### LIME ##############################\n')
    lime = Lime(
        forward_func=model,
        interpretable_model=SkLearnLinearRegression(),  # Surrogate model
        similarity_func=None  # Use default (exponential kernel)
    )

    for i in range(samples.shape[0]):
        sample = samples[i].unsqueeze(0)  # shape [1, features]
        label = labels[i].item()

        sample_np = sample.squeeze().cpu().numpy().reshape(28, 28)
        sample_rgb = np.stack([sample_np]*3, axis=-1)

        segments = slic(sample_rgb, n_segments=100, compactness=1)
        segments = np.unique(segments, return_inverse=True)[1].reshape(segments.shape) # make segments start with 0

        feature_mask = torch.tensor(segments, dtype=torch.long).view(-1).unsqueeze(0).to(device)

        attributions = lime.attribute(sample, target=label, n_samples=1500, feature_mask=feature_mask)
        attr = attributions.squeeze().cpu().numpy()

        heatmap = np.zeros_like(segments, dtype=np.float32)
        for seg_val in np.unique(segments):
            heatmap[segments == seg_val] = attr[seg_val]

        plot_3_part_heatmap(sample_rgb, segments, heatmap, title=f'LIME\ny_true: {labels[i]}, y_pred: {preds[i]}')

if __name__ == '__main__':
    main()
