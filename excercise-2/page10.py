import os
os.environ["OMP_NUM_THREADS"] = '1'

import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import torch
import torch.nn as nn
import torch.optim as optim

OUT_DIR = 'page4'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_DBSCAN(x, eps):
    algo = DBSCAN(eps=eps)
    algo.fit(x)
    y_pred = algo.labels_.astype(int)
    n_clusters = len(set(y_pred))
    return y_pred, n_clusters

def compute_KMeans(x, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    y_pred = kmeans.fit_predict(x)
    return y_pred

def evaluate(true_labels, predicted_labels, x):
    ari = adjusted_rand_score(true_labels, predicted_labels)
    hom = homogeneity_score(true_labels, predicted_labels)
    comp = completeness_score(true_labels, predicted_labels)
    sil = silhouette_score(x, predicted_labels, metric='euclidean')
    return ari, hom, comp, sil

def choose_dataset():
    print("Choose a dataset:")
    print("1 - Iris")
    print("2 - Wine")
    print("3 - Breast Cancer Wisconsin")
    choice = input("Enter number (1/2/3): ")

    if choice == '1':
        data = load_iris()
        name = "Iris"
    elif choice == '2':
        data = load_wine()
        name = "Wine"
    elif choice == '3':
        data = load_breast_cancer()
        name = "Breast Cancer Wisconsin"
    else:
        print("Invalid choice. Defaulting to Iris.")
        data = load_iris()
        name = "Iris"

    return data.data, data.target, name

def train_knn(X_train, y_train, X_test, y_test, neighbors_range):
    accuracies_train = []
    accuracies_test = []

    for n_neighbors in neighbors_range:
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train, y_train)

        acc_train = accuracy_score(y_train, knn.predict(X_train))
        acc_test = accuracy_score(y_test, knn.predict(X_test))

        accuracies_train.append(acc_train)
        accuracies_test.append(acc_test)

    return accuracies_train, accuracies_test

def run_svm(X_train, y_train, X_test, y_test, c_values):
    accuracies_train = []
    accuracies_test = []

    for c in c_values:
        model = SVC(kernel='rbf', C=c)
        model.fit(X_train, y_train)

        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)

        accuracies_train.append(train_acc)
        accuracies_test.append(test_acc)

    return accuracies_train, accuracies_test

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, activation):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)

def train_mlp(X_train, y_train, X_test, y_test, hidden_size, epochs=200, activation_name='ReLU'):
    # Ensure X_train is a tensor
    X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_train, dtype=torch.long).to(device)

    X_tensor_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_tensor_test = torch.tensor(y_test, dtype=torch.long).to(device)

    input_dim = X_train.shape[len(X_train.shape)-1]    
    output_dim = y_train.shape[len(y_train.shape)-1]

    model = MLP(input_dim, output_dim, hidden_size, nn.ReLU()).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = loss_fn(output, y_tensor)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        preds = model(X_tensor).argmax(dim=1)
        acc_train = (preds == y_tensor).float().mean().item()

        preds = model(X_tensor_test).argmax(dim=1)
        acc_test = (preds == y_tensor_test).float().mean().item()

    return model, acc_train, acc_test

def main():
    X, y_true, dataset_name = choose_dataset()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_true, test_size=0.2, random_state=42, #stratify=labels
    )

    print(f"\n=== Running on {dataset_name} dataset ===")

    print("\n=== KNN ===")

    nbors_range = range(1, 10)

    accuracies_train, accuracies_test = train_knn(X_train, y_train, X_test, y_test, nbors_range)

    for i, _ in enumerate(accuracies_train):
        nbrs = nbors_range[i]
        print(f'KNN nbrs={nbrs}: train: {accuracies_train[i]:.4f} test: {accuracies_test[i]:.4f}')

    print("\n=== SVM ===")

    c_range = np.logspace(-2, 6, 9)

    accuracies_train, accuracies_test = run_svm(X_train, y_train, X_test, y_test, c_range)

    for i, _ in enumerate(accuracies_train):
        c = c_range[i]
        print(f'SVM c={c}: train: {accuracies_train[i]:.4f} test: {accuracies_test[i]:.4f}')

    print("\n=== MLP ===")

    hidden_range = range(1, 10)

    for hsize in hidden_range:
        _, train_acc, test_acc = train_mlp(X_train, y_train, X_test, y_test, hidden_size=hsize, epochs=100, activation_name='ReLU')
        print(f'MLP hidden_size={hsize}: train: {train_acc:.4f} test: {test_acc:.4f}')

if __name__ == "__main__":
    main()
