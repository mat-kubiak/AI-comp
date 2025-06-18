import torch
from src.mlp import NN
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.preprocessing import StandardScaler
import numpy as np

from captum.attr import IntegratedGradients
from captum.attr import Lime
from captum._utils.models.linear_model import SkLearnLinearRegression, SkLearnLasso

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

if __name__ == '__main__':
    main()
