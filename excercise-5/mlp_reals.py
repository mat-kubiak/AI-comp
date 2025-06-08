import torch
from mlp import NN
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.preprocessing import StandardScaler
from captum.attr import IntegratedGradients
import numpy as np
from lime.lime_tabular import LimeTabularExplainer

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
    ig = IntegratedGradients(model)
    attributions, delta = ig.attribute(samples, target=labels, return_convergence_delta=True)
    print("Attributions:", attributions)

    # LIME
    def predict_fn(input_np):
        input_tensor = torch.tensor(input_np, dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)
        return probs.cpu().numpy()

    explainer = LimeTabularExplainer(X, 
                                     feature_names=dataset.feature_names,
                                     class_names=dataset.target_names,
                                     discretize_continuous=True,
                                     mode='classification')

    print("\nLIME Explanations (first 3 samples):")
    for i in range(min(3, num_samples)):
        explanation = explainer.explain_instance(X[random_indices[i]],
                                                 predict_fn,
                                                 num_features=5)
        print(f"\nSample index: {random_indices[i]}")
        # explanation.show_in_notebook(show_table=True)  # works in Jupyter
        print(explanation.as_list())

if __name__ == '__main__':
    main()