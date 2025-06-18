import torch
from matplotlib import pyplot as plt

from mlp import NN
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.preprocessing import StandardScaler
import numpy as np

from captum.attr import IntegratedGradients
from captum.attr import Lime
from captum._utils.models.linear_model import SkLearnLinearRegression, SkLearnLasso

def plot_pdp(
    model,
    X,
    feature_index=0,
    target_class=0,
    num_points=50,
    device='cpu',
    feature_names=None,
    class_names=None
):
    import numpy as np
    import matplotlib.pyplot as plt
    import torch

    # Generate values for the selected feature
    feature_vals = np.linspace(X[:, feature_index].min(), X[:, feature_index].max(), num_points)

    # Use mean of other features
    baseline = np.mean(X, axis=0)
    pdp_preds = []

    for val in feature_vals:
        sample = np.array(baseline, copy=True)
        sample[feature_index] = val
        sample_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(sample_tensor)
            prob = torch.softmax(output, dim=1)[0, target_class].item()
        pdp_preds.append(prob)

    # Use proper names if provided
    feature_label = feature_names[feature_index] if feature_names is not None else f'Feature {feature_index}'
    class_label = class_names[target_class] if class_names is not None else f'Class {target_class}'

    # Plotting
    plt.figure(figsize=(6, 4))
    plt.plot(feature_vals, pdp_preds, label=class_label)
    plt.xlabel(feature_label)
    plt.ylabel('Prawdopodobieństwo przewidzenia')
    plt.title(f'Partial Dependence Plot dla {class_label}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_feature_importance_from_pdp(model, X, target_class=0, num_points=50, device='cpu', feature_names=None, class_names=None):
    num_features = X.shape[1]
    baseline = np.mean(X, axis=0)

    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(num_features)]

    importances = []

    for i in range(num_features):
        feature_vals = np.linspace(X[:, i].min(), X[:, i].max(), num_points)
        pdp_preds = []

        for val in feature_vals:
            sample = np.array(baseline, copy=True)
            sample[i] = val
            sample_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(sample_tensor)
                prob = torch.softmax(output, dim=1)[0, target_class].item()
            pdp_preds.append(prob)

        importance = max(pdp_preds) - min(pdp_preds)
        importances.append(importance)

    # Sort by importance
    sorted_indices = np.argsort(importances)
    sorted_features = [feature_names[i] for i in sorted_indices]
    sorted_importances = [importances[i] for i in sorted_indices]

    class_label = class_names[target_class] if class_names is not None else f'Class {target_class}'

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.scatter(sorted_importances, sorted_features, color='blue')
    plt.xlabel('Szacowana ważność cechy (Zasięg (PDP)')
    plt.ylabel('Cecha')
    plt.title(f'Feature Importance via PDP (Klasa {class_label})')
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def choose_dataset():
    print("Choose a dataset:")
    print("1 - Iris")
    print("2 - Wine")
    print("3 - Breast Cancer Wisconsin")
    choice = input("Enter number (1/2/3): ")

    if choice == '1':
        model_name = 'iris_best_80.ckpt'
        dataset = load_iris()
    elif choice == '2':
        model_name = 'wine_best_97.ckpt'
        dataset = load_wine()
    elif choice == '3':
        model_name = 'bcw_best_95.ckpt'
        dataset = load_breast_cancer()
    else:
        print("Invalid choice. Defaulting to Iris.")
        model_name = 'iris_best_80.ckpt'
        dataset = load_iris()

    return model_name, dataset

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_name, dataset = choose_dataset()

    feature_names = dataset.feature_names  # or manually define
    class_names = dataset.target_names

    # Load model
    model = NN.load_from_checkpoint(f'checkpoints/{model_name}')
    model.to(device)
    model.eval()

    # Unpack data
    X = dataset.data
    y = dataset.target

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Choose 5 random sample indices without replacement
    num_samples = 20
    random_indices = np.random.choice(len(X), size=num_samples, replace=False)

    # Select samples and labels at those indices
    samples = torch.tensor(X[random_indices], dtype=torch.float32).to(device)
    labels = torch.tensor(y[random_indices], dtype=torch.long).to(device)

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
    print("Attributions:", attributions)

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
        attributions = lime.attribute(sample, target=label, n_samples=50)

        print(f"y_true: {label}, y_pred: {preds[i].item()},\nsample: {sample},\nattr:   {attributions}\n")

    for target_class in range(len(class_names)):
        plot_pdp(model, X, feature_index=1, target_class=target_class, num_points=50, device=device,
                 feature_names=feature_names, class_names=class_names)
        plot_feature_importance_from_pdp(model, X, target_class=target_class, num_points=50, device=device,
                                         feature_names=feature_names, class_names=class_names)

if __name__ == '__main__':
    main()
