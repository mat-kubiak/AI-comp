import pandas as pd
import utils
import torch
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

OUT_DIR = 'page8'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


# Define your MLP model class
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


# Function to train MLP and return accuracy over epochs
def train_mlp(X_train, y_train, X_test, y_test, hidden_size, num_epochs):
    X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_train, dtype=torch.long).to(device)

    model = MLP(2, hidden_size, nn.ReLU()).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    accuracies_train_epoch = []
    accuracies_test_epoch = []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = loss_fn(output, y_tensor)
        loss.backward()
        optimizer.step()

        # Training accuracy for this epoch
        with torch.no_grad():
            train_preds = model(X_tensor).argmax(dim=1)
            train_acc = (train_preds == y_tensor).float().mean().item()
            accuracies_train_epoch.append(train_acc)

            # Test accuracy for this epoch
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
            y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
            test_preds = model(X_test_tensor).argmax(dim=1)
            test_acc = (test_preds == y_test_tensor).float().mean().item()
            accuracies_test_epoch.append(test_acc)

    return model, accuracies_train_epoch, accuracies_test_epoch


# Function for running the experiment multiple times and collecting results
def run_multiple_trainings(X_train, y_train, X_test, y_test, dataset_name, num_epochs, hidden_size=41, num_runs=10):
    results = []

    for run in range(num_runs):
        model, accuracies_train_epoch, accuracies_test_epoch = train_mlp(X_train, y_train, X_test, y_test, hidden_size, num_epochs)

        # Get accuracies for the first, best, and last epochs
        first_epoch_acc_train = accuracies_train_epoch[0]
        first_epoch_acc_test = accuracies_test_epoch[0]

        best_epoch = np.argmax(accuracies_test_epoch)
        best_epoch_acc_train = accuracies_train_epoch[best_epoch]
        best_epoch_acc_test = accuracies_test_epoch[best_epoch]

        last_epoch_acc_train = accuracies_train_epoch[-1]
        last_epoch_acc_test = accuracies_test_epoch[-1]

        # Save the results
        results.append({
            'Run': run + 1,
            'First Epoch Train Accuracy': first_epoch_acc_train,
            'First Epoch Test Accuracy': first_epoch_acc_test,
            'Best Epoch': best_epoch + 1,
            'Best Epoch Train Accuracy': best_epoch_acc_train,
            'Best Epoch Test Accuracy': best_epoch_acc_test,
            'Last Epoch Train Accuracy': last_epoch_acc_train,
            'Last Epoch Test Accuracy': last_epoch_acc_test
        })

        # Plot Decision Boundary for the first, best, and last epochs
        # First Epoch
        plot_decision_boundary(model, X_train, y_train, f"Decision Boundary - First Epoch ({dataset_name} - Run {run + 1})")
        plt.savefig(f"{OUT_DIR}/decision_boundary_first_epoch_run{run + 1}_{dataset_name}.png")
        plt.close()

        # Best Epoch (highest test accuracy)
        plot_decision_boundary(model, X_train, y_train, f"Decision Boundary - Best Epoch ({dataset_name} - Run {run + 1})")
        plt.savefig(f"{OUT_DIR}/decision_boundary_best_epoch_run{run + 1}_{dataset_name}.png")
        plt.close()

        # Last Epoch
        plot_decision_boundary(model, X_train, y_train, f"Decision Boundary - Last Epoch ({dataset_name} - Run {run + 1})")
        plt.savefig(f"{OUT_DIR}/decision_boundary_last_epoch_run{run + 1}_{dataset_name}.png")
        plt.close()

    # Convert results into a DataFrame for easy tabulation
    df_results = pd.DataFrame(results)

    # Save the table as a CSV file
    df_results.to_csv(f"{OUT_DIR}/results_{dataset_name}.csv", index=False)

    # Print the results as a table
    print(df_results)

    # Optionally, plot the accuracy over epochs for the last run (or any other)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), accuracies_train_epoch, label='Train Accuracy', color='b')
    plt.plot(range(1, num_epochs + 1), accuracies_test_epoch, label='Test Accuracy', color='r')
    plt.title(f"Accuracy Over Epochs ({dataset_name})", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{OUT_DIR}/accuracy_over_epochs_{dataset_name}.png")
    plt.close()


# Function to run the MLP experiment
def run_mlp_experiment(X_train, y_train, X_test, y_test, dataset_name, num_epochs=10, hidden_size=41):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Run the experiment 10 times with different initializations
    run_multiple_trainings(X_train_scaled, y_train, X_test_scaled, y_test, dataset_name, num_epochs, hidden_size)


# Example of using the function with your datasets
for filename in utils.files[2:]:
    data, labels = utils.load('./data/' + filename)

    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, train_size=0.8, random_state=42, stratify=labels
    )

    run_mlp_experiment(X_train, y_train, X_test, y_test, filename)
