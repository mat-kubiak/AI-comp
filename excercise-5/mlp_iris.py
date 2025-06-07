import torch
from mlp import NN
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from captum.attr import IntegratedGradients
import numpy as np

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = NN.load_from_checkpoint('checkpoints/iris_best_80.ckpt')
    model.to(device)
    model.eval()

    # Load iris data
    iris = load_iris()
    X = iris.data
    y = iris.target

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
    ig = IntegratedGradients(model)
    attributions, delta = ig.attribute(samples, target=labels, return_convergence_delta=True)

    print("Attributions:", attributions)

if __name__ == '__main__':
    main()