import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import utils

OUT_DIR = 'page5'

def train_knn(X_train, y_train, X_test, y_test, min_neighbors, max_neighbors):
    accuracies_train = []
    accuracies_test = []
    best_acc = 0

    cm_train_best = None
    cm_test_best = None

    for n_neighbors in range(min_neighbors, max_neighbors + 1):
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train, y_train)

        acc_train = accuracy_score(y_train, knn.predict(X_train))
        acc_test = accuracy_score(y_test, knn.predict(X_test))

        accuracies_train.append(acc_train)
        accuracies_test.append(acc_test)

        cm_train = confusion_matrix(y_train, knn.predict(X_train))
        cm_test = confusion_matrix(y_test, knn.predict(X_test))

        if acc_test > best_acc:
            best_acc = acc_test
            cm_train_best = cm_train
            cm_test_best = cm_test

    return accuracies_train, accuracies_test, cm_train_best, cm_test_best


def run_knn_experiment(X_train, y_train, X_test, y_test, dataset_name, min_neighbors=1, max_neighbors=20):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the KNN models
    accuracies_train, accuracies_test, cm_train_best, cm_test_best = train_knn(
        X_train_scaled, y_train, X_test_scaled, y_test, min_neighbors, max_neighbors)


    best_acc_idx = np.argmax(accuracies_test)  # Index of highest test accuracy (biggest)
    worst_acc_idx = np.argmin(accuracies_test)  # Index of lowest test accuracy (worst)

    best_n_neighbors = best_acc_idx + min_neighbors  # Biggest neighbors based on highest accuracy
    worst_n_neighbors = worst_acc_idx + min_neighbors  # Worst neighbors based on lowest accuracy

    # Plot accuracy vs number of neighbors
    plt.figure(figsize=(10, 6))
    plt.plot(range(min_neighbors, max_neighbors + 1), accuracies_train, label='Train Accuracy', color='b', marker='o')
    plt.plot(range(min_neighbors, max_neighbors + 1), accuracies_test, label='Test Accuracy', color='r', marker='x')
    plt.axvline(x=best_n_neighbors, color='g', linestyle='--', label=f'Best n_neighbors: {best_n_neighbors}')
    plt.axvline(x=worst_n_neighbors, color='r', linestyle='--', label=f'Worst n_neighbors: {worst_n_neighbors}')
    plt.title(f"KNN Accuracy vs. Number of Neighbors ({dataset_name})", fontsize=14)
    plt.xlabel("Number of Neighbors", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{OUT_DIR}/accuracy_vs_neighbors_{dataset_name}.png")
    plt.close()

    worst_knn_model = KNeighborsClassifier(n_neighbors=worst_n_neighbors)
    worst_knn_model.fit(X_train, y_train)

    best_knn_model = KNeighborsClassifier(n_neighbors=best_n_neighbors)
    best_knn_model.fit(X_train, y_train)

    # Decision boundary for the worst accuracy (lowest number of neighbors)
    utils.plot_decision_boundary(worst_knn_model, X_train, y_train,
                           f"Decision Boundary - Worst Accuracy ({dataset_name})")
    plt.savefig(f"{OUT_DIR}/decision_boundary_worst_accuracy_{dataset_name}.png")
    plt.close()


    # Decision boundary for the best accuracy (based on test set)
    utils.plot_decision_boundary(best_knn_model, X_train, y_train, f"Decision Boundary - Best Accuracy ({dataset_name})")
    plt.savefig(f"{OUT_DIR}/decision_boundary_best_accuracy_{dataset_name}.png")
    plt.close()


    # Confusion matrix for the worst accuracy
    utils.plot_confusion_matrix(confusion_matrix(y_train, worst_knn_model.predict(X_train)),
                          classes=np.unique(y_train),
                          title=f"Confusion Matrix - Worst Accuracy ({dataset_name})")
    plt.savefig(f"{OUT_DIR}/confusion_matrix_worst_accuracy_{dataset_name}_train.png")
    plt.close()


    # Confusion matrix for the best accuracy
    utils.plot_confusion_matrix(confusion_matrix(y_train, best_knn_model.predict(X_train)),
                          classes=np.unique(y_train),
                          title=f"Confusion Matrix - Best Accuracy ({dataset_name})")
    plt.savefig(f"{OUT_DIR}/confusion_matrix_best_accuracy_{dataset_name}_train.png")
    plt.close()




for filename in utils.files[1:]:
    data, labels = utils.load('./data/' + filename)

    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, train_size=0.2, random_state=42, stratify=labels
    )

    run_knn_experiment(X_train, y_train, X_test, y_test, filename)