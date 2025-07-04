import torch
from mlp import NN

from extraction import ext_methods
import torchvision.datasets as datasets

from sklearn.preprocessing import StandardScaler
import numpy as np

from captum.attr import IntegratedGradients
from captum.attr import Lime
from captum._utils.models.linear_model import SkLearnLinearRegression, SkLearnLasso

import matplotlib.pyplot as plt

from skimage.segmentation import slic

def choose_dataset():
    print("Choose a feature extracton method:")
    print("1 - flatten")
    # print("2 - cnn")
    choice = input("Enter number (1): ")

    if choice == '1':
        model_name = 'mnist_flatten_best_92.ckpt'
        dataset = datasets.MNIST(root='./datasets', train=False, transform=ext_methods['flatten'], download=True)
    # elif choice == '2':
    #     model_name = 'mnist_edges_all_best_46.ckpt'
    #     method = 'edges_all'
    else:
        print("Invalid choice. Defaulting to flatten.")
        model_name = 'mnist_flatten_best_92.ckpt'
        dataset = datasets.MNIST(root='./datasets', train=False, transform=ext_methods['flatten'], download=True)

    return model_name, dataset

def plot_heatmap(data, title=''):
    max_val = np.abs(data).max()

    fig, ax = plt.subplots()
    im = ax.imshow(data, vmin=-max_val, vmax=max_val, cmap='bwr')
    if title != '':
        ax.set_title(title)
    fig.colorbar(im, ax=ax)
    plt.show()

def shuffle_by_value(data):
    unique_data = np.unique(data)
    shuffled_labels = np.random.permutation(unique_data)

    label_map = dict(zip(unique_data, shuffled_labels))

    return np.vectorize(label_map.get)(data)

def plot_2_part_heatmap(original_data, attr_data, title=''):
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

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_name, dataset = choose_dataset()

    # Load model
    model = NN.load_from_checkpoint(f'checkpoints/{model_name}')
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

    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)

    # Choose 5 random sample indices without replacement
    num_samples = 5
    random_indices = np.random.choice(len(X), size=num_samples, replace=False)

    # Select samples and labels at those indices
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
    labels = labels.to(torch.long)  # or labels.long()
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
