import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import torch
import torch.nn as nn
import torch.optim as optim

import utils

# Set seeds for reproducibility
np.random.seed(0)
torch.manual_seed(0)

OUT_DIR = 'page1'
KERNELS = ['linear', 'rbf']
C_VALUES = [0.1, 1, 10, 100]
ACTIVATIONS = {'relu': nn.ReLU(),  'identity': nn.Identity()}
HIDDEN_SIZES = [5, 10, 20, 50]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def plot_decision_boundary(model_fn, X, y, title):
    # Create a meshgrid over the input space
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))

    # Predict over the grid
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model_fn(grid).reshape(xx.shape)

    # Plot decision boundary and data points
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.title(title)

# === 1. SVM Experiment ===
def run_svm_experiment(X_train, y_train, X_test, y_test, dataset_name):
    best_acc = -1
    best_model = None
    best_params = ()

    for kernel in KERNELS:
        for C in C_VALUES:
            model = SVC(kernel=kernel, C=C)
            model.fit(X_train, y_train)
            train_acc = model.score(X_train, y_train)

            if train_acc > best_acc:
                best_acc = train_acc
                best_model = model
                best_params = (kernel, C)

    kernel, C = best_params
    print(f"[{dataset_name}] Best SVM: kernel={kernel}, C={C}, train_acc={best_acc:.2f}")

    # Visualization
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plot_decision_boundary(lambda x: best_model.predict(x), X_train, y_train,
                           f"SVM Train | {kernel}, C={C}")
    plt.subplot(1, 2, 2)
    plot_decision_boundary(lambda x: best_model.predict(x), X_test, y_test,
                           f"SVM Test | {kernel}, C={C}")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/svm_{filename[0:-4]}.png")
    plt.close()

# === 2. MLP Experiment (PyTorch) ===
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x):
        return self.model(x)

def train_mlp(X_train, y_train, hidden_size, activation_name):
    X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_train, dtype=torch.long).to(device)

    model = MLP(2, hidden_size, ACTIVATIONS[activation_name]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = loss_fn(output, y_tensor)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        preds = model(X_tensor).argmax(dim=1)
        acc = (preds == y_tensor).float().mean().item()
    return model, acc

def run_mlp_experiment(X_train, y_train, X_test, y_test, dataset_name):
    best_acc = -1
    best_params = ()

    for hidden_size in HIDDEN_SIZES:
        for act_name in ACTIVATIONS:
            model, acc = train_mlp(X_train, y_train, hidden_size, act_name)
            if acc > best_acc:
                best_acc = acc
                best_params = (hidden_size, act_name)

    hidden_size, act_name = best_params
    print(f"[{dataset_name}] Best MLP: activation={act_name}, hidden={hidden_size}, train_acc={best_acc:.2f}")

    def model_fn(x):
        model.eval()
        x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
        with torch.no_grad():
            preds = model(x_tensor).argmax(dim=1).cpu().numpy()
        return preds

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plot_decision_boundary(model_fn, X_train, y_train,
                           f"MLP Train | {act_name}, h={hidden_size}")
    plt.subplot(1, 2, 2)
    plot_decision_boundary(model_fn, X_test, y_test,
                           f"MLP Test | {act_name}, h={hidden_size}")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/mlp_{filename[0:-4]}.png")
    plt.close()

# === Run all experiments ===

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

for filename in utils.files:
    data, labels = utils.load('./data/' + filename)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Standardize the features using StandardScaler
    scaler = StandardScaler()

    # Fit the scaler on the training data and transform both the train and test data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Run the SVM experiment on the standardized data
    run_svm_experiment(X_train_scaled, y_train, X_test_scaled, y_test, filename)

    # Run the MLP experiment on the standardized data
    run_mlp_experiment(X_train_scaled, y_train, X_test_scaled, y_test, filename)
