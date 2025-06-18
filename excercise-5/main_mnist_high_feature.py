import torch
import numpy as np

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from captum.attr import IntegratedGradients
from captum.attr import Lime
from captum._utils.models.linear_model import SkLearnLinearRegression

from skimage.segmentation import slic

from src.mlp import NN
from src.cnn_mnist import CNN
from src.utils import plot_2_part_heatmap, plot_3_part_heatmap

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

    # Integrated Gradients
    print('\n### Integrated Gradents ###############\n')
    ig = IntegratedGradients(model)
    attributions, delta = ig.attribute(samples, target=labels, return_convergence_delta=True)

    attributions = np.reshape(attributions.cpu(), (NUM_SAMPLES, 28, 28))
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
        sample = samples[i].unsqueeze(0)
        label = labels[i].item()

        sample_np = sample.squeeze().cpu().numpy()
        if len(sample_np.shape) == 1:
            sample_np = sample_np.reshape((28, 28))
        sample_rgb = np.stack([sample_np]*3, axis=-1)

        segments = slic(sample_rgb, n_segments=100, compactness=1)
        segments = segments - np.min(segments) # make segments start with 0 to shut down warning

        feature_mask = torch.tensor(segments, dtype=torch.long).unsqueeze(0).to(device) # segments [28, 28] -> [1, 28, 28]

        # for mlp, the shapes must be:
        # sample: (1, 784)
        # feature_mask: (784)
        if len(sample.shape) == 2:
            feature_mask = feature_mask.view(-1)

        attributions = lime.attribute(sample, target=label, n_samples=1500, feature_mask=feature_mask)
        attr = attributions.squeeze().cpu().numpy()

        # ensure correct shape for mlp
        attr = attr.reshape((28, 28))

        heatmap = np.zeros_like(segments, dtype=np.float32)
        for seg_val in np.unique(segments):
            mask = segments == seg_val
            heatmap[mask] = attr[mask].mean()

        plot_3_part_heatmap(sample_rgb, segments, heatmap, title=f'LIME\ny_true: {labels[i]}, y_pred: {preds[i]}')

if __name__ == '__main__':
    main()
