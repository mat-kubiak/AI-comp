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
    attributions, delta = ig.attribute(samples, target=labels, return_convergence_delta=True)

    attributions = np.reshape(attributions.cpu(), (5, 28, 28))
    for i in range(len(attributions)):
        plot_heatmap(attributions[i], f'Integrated Gradients\ny_true: {labels[i]}, y_pred: {preds[i]}')

    # LIME
    # print('\n### LIME ##############################\n')
    # lime = Lime(
    #     forward_func=model,
    #     interpretable_model=SkLearnLinearRegression(),  # Surrogate model
    #     similarity_func=None  # Use default (exponential kernel)
    # )

    # for i in range(samples.shape[0]):
    #     sample = samples[i].unsqueeze(0)  # shape [1, features]
    #     label = labels[i].item()
    #     attributions = lime.attribute(sample, target=label, n_samples=2000)

    #     plot_heatmap(np.reshape(attributions.cpu(), (28, 28)), f'LIME\ny_true: {labels[i]}, y_pred: {preds[i]}')

if __name__ == '__main__':
    main()
