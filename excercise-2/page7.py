import utils
from matplotlib import pyplot as plt
from pyomo.contrib.parmest.graphics import sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim

OUT_DIR = 'page7'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def plot_confusion_matrix(cm, classes, title):
    # Plot confusion matrix using Seaborn heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)


def plot_decision_boundary(model, X, y, title):
    # Create a meshgrid over the input space
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))

    # Predict over the grid by creating a callable function
    grid = np.c_[xx.ravel(), yy.ravel()]

    model.eval()
    grid_tensor = torch.tensor(grid, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = model(grid_tensor).argmax(dim=1).cpu().numpy()

    # Reshape the output back to the grid shape
    Z = preds.reshape(xx.shape)

    # Plot decision boundary and data points
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.title(title)



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


def train_mlp(X_train, y_train, hidden_size, activation_name='ReLU'):
    # Ensure X_train is a tensor
    X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_train, dtype=torch.long).to(device)

    model = MLP(2, hidden_size,  nn.ReLU()).to(device)
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
    scaler = StandardScaler()

    # Fit the scaler to the training data and transform both the train and test data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Hidden layer sizes between 1 and 60
    hidden_sizes = range(1, 61)  # Hidden sizes from 1 to 60

    accuracies_train = []
    accuracies_test = []

    for hidden_size in hidden_sizes:
        # Train MLP model for each hidden size
        model, train_acc = train_mlp(X_train_scaled, y_train, hidden_size)

        # Test accuracy on the test set
        model.eval()
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

        with torch.no_grad():
            test_preds = model(X_test_tensor).argmax(dim=1)
            test_acc = (test_preds == y_test_tensor).float().mean().item()

        accuracies_train.append(train_acc)
        accuracies_test.append(test_acc)

    # Identify the best, worst, and biggest accuracy cases (based on test accuracy)
    best_acc_idx = np.argmax(accuracies_test)  # Index of highest test accuracy (best)
    worst_acc_idx = np.argmin(accuracies_test)  # Index of lowest test accuracy (worst)

    best_hidden_size = hidden_sizes[best_acc_idx]  # Best hidden size based on highest accuracy
    worst_hidden_size = hidden_sizes[worst_acc_idx]  # Worst hidden size based on lowest accuracy

    # Plot accuracy vs hidden layer sizes
    plt.figure(figsize=(10, 6))
    plt.plot(hidden_sizes, accuracies_train, label='Train Accuracy', color='b', marker='o')
    plt.plot(hidden_sizes, accuracies_test, label='Test Accuracy', color='r', marker='x')
    plt.axvline(x=best_hidden_size, color='g', linestyle='--', label=f'Best hidden_size: {best_hidden_size}')
    plt.axvline(x=worst_hidden_size, color='r', linestyle='--', label=f'Worst hidden_size: {worst_hidden_size}')
    plt.title(f"MLP Accuracy vs. Hidden Layer Size ({dataset_name})", fontsize=14)
    plt.xlabel("Hidden Layer Size", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{OUT_DIR}/accuracy_vs_hidden_size_{dataset_name}.png")
    plt.close()

    # Train the MLP models with best and worst hidden layer sizes
    best_mlp_model, _ = train_mlp(X_train_scaled, y_train, best_hidden_size)
    worst_mlp_model, _ = train_mlp(X_train_scaled, y_train, worst_hidden_size)

    # Confusion matrix for the worst accuracy
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    worst_preds = worst_mlp_model(X_train_tensor).argmax(dim=1).cpu().numpy()
    plot_confusion_matrix(confusion_matrix(y_train, worst_preds),
                          classes=np.unique(y_train),
                          title=f"Confusion Matrix - Worst Accuracy ({dataset_name})")
    plt.savefig(f"{OUT_DIR}/confusion_matrix_worst_accuracy_{dataset_name}_train.png")
    plt.close()

    # Confusion matrix for the best accuracy
    best_preds = best_mlp_model(X_train_tensor).argmax(dim=1).cpu().numpy()
    plot_confusion_matrix(confusion_matrix(y_train, best_preds),
                          classes=np.unique(y_train),
                          title=f"Confusion Matrix - Best Accuracy ({dataset_name})")
    plt.savefig(f"{OUT_DIR}/confusion_matrix_best_accuracy_{dataset_name}_train.png")
    plt.close()

    # Decision boundary for the worst accuracy (lowest hidden size)
    plot_decision_boundary(worst_mlp_model, X_train_scaled, y_train, f"Decision Boundary - Worst Accuracy ({dataset_name})")
    plt.savefig(f"{OUT_DIR}/decision_boundary_worst_accuracy_{dataset_name}.png")
    plt.close()

    # Decision boundary for the best accuracy (highest hidden size)
    plot_decision_boundary(best_mlp_model, X_train_scaled, y_train, f"Decision Boundary - Best Accuracy ({dataset_name})")
    plt.savefig(f"{OUT_DIR}/decision_boundary_best_accuracy_{dataset_name}.png")
    plt.close()


for filename in utils.files[1:]:
    data, labels = utils.load('./data/' + filename)

    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.8, random_state=42, stratify=labels
    )

    run_mlp_experiment(X_train, y_train, X_test, y_test, filename)