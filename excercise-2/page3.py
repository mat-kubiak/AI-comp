import utils
from matplotlib import pyplot as plt
from pyomo.contrib.parmest.graphics import sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

# New C values will be exponentially spaced
C_VALUES = np.logspace(-2, 6, 9)
OUT_DIR = 'page3'


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
    Z = model.predict(grid)  # Call predict method on the model

    # Reshape the output back to the grid shape
    Z = Z.reshape(xx.shape)

    # Plot decision boundary and data points
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.title(title)



# --- SVM Experiment ---
def run_svm(X_train, y_train, X_test, y_tests):
    accuracies_train = []
    accuracies_test = []
    cm_train_best = None
    cm_test_best = None

    for C in C_VALUES:
        model = SVC(kernel='rbf', C=C)
        model.fit(X_train, y_train)

        # Train and test accuracies
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)

        # Confusion matrices
        cm_train = confusion_matrix(y_train, model.predict(X_train))
        cm_test = confusion_matrix(y_test, model.predict(X_test))

        # Store results for each C
        accuracies_train.append(train_acc)
        accuracies_test.append(test_acc)

        # If it's the first run, initialize confusion matrices
        if cm_train_best is None:
            cm_train_best = cm_train
            cm_test_best = cm_test

    # Return the requested results
    return accuracies_train, accuracies_test, cm_train_best, cm_test_best


def run_svm_experiment(X_train, y_train, X_test, y_test, dataset_name):
    scaler = StandardScaler()

    # Fit the scaler to the training data and transform both the train and test data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the SVM models
    accuracies_train, accuracies_test, cm_train_best, cm_test_best = run_svm(
        X_train_scaled, y_train, X_test_scaled, y_test)

    # Identify the best, worst, and biggest accuracy cases (based on test accuracy)
    best_acc_idx = np.argmax(accuracies_test)  # Index of highest test accuracy (best)
    worst_acc_idx = np.argmin(accuracies_test)  # Index of lowest test accuracy (worst)

    best_C = C_VALUES[best_acc_idx]  # Best C value based on highest accuracy
    worst_C = C_VALUES[worst_acc_idx]  # Worst C value based on lowest accuracy

    # Calculate log(C) values
    log_C_VALUES = np.log10(C_VALUES)

    # Plot accuracy vs log(C) values
    plt.figure(figsize=(10, 6))
    plt.plot(log_C_VALUES, accuracies_train, label='Train Accuracy', color='b', marker='o')
    plt.plot(log_C_VALUES, accuracies_test, label='Test Accuracy', color='r', marker='x')
    plt.axvline(x=np.log10(best_C), color='g', linestyle='--', label=f'Best log(C): {np.log10(best_C):.2f}')
    plt.axvline(x=np.log10(worst_C), color='r', linestyle='--', label=f'Worst log(C): {np.log10(worst_C):.2f}')
    plt.title(f"SVM Accuracy vs. log(C) ({dataset_name})", fontsize=14)
    plt.xlabel("log(C) Value", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{OUT_DIR}/accuracy_vs_logC_{dataset_name}.png")
    plt.close()

    # Train the SVM models with best and worst C values
    best_svm_model = SVC(kernel='rbf', C=best_C)
    best_svm_model.fit(X_train_scaled, y_train)

    worst_svm_model = SVC(kernel='rbf', C=worst_C)
    worst_svm_model.fit(X_train_scaled, y_train)

    # Confusion matrix for the worst accuracy
    plot_confusion_matrix(confusion_matrix(y_train, worst_svm_model.predict(X_train)),
                          classes=np.unique(y_train),
                          title=f"Confusion Matrix - Worst Accuracy ({dataset_name})")
    plt.savefig(f"{OUT_DIR}/confusion_matrix_worst_accuracy_{dataset_name}_train.png")
    plt.close()

    # Confusion matrix for the best accuracy
    plot_confusion_matrix(confusion_matrix(y_train, best_svm_model.predict(X_train)),
                          classes=np.unique(y_train),
                          title=f"Confusion Matrix - Best Accuracy ({dataset_name})")
    plt.savefig(f"{OUT_DIR}/confusion_matrix_best_accuracy_{dataset_name}_train.png")
    plt.close()

    # Decision boundary for the worst accuracy (lowest C)
    plot_decision_boundary(worst_svm_model, X_train_scaled, y_train,
                           f"Decision Boundary - Worst Accuracy ({dataset_name})")
    plt.savefig(f"{OUT_DIR}/decision_boundary_worst_accuracy_{dataset_name}.png")
    plt.close()

    # Decision boundary for the best accuracy (highest C)
    plot_decision_boundary(best_svm_model, X_train_scaled, y_train, f"Decision Boundary - Best Accuracy ({dataset_name})")
    plt.savefig(f"{OUT_DIR}/decision_boundary_best_accuracy_{dataset_name}.png")
    plt.close()




for filename in utils.files[1:]:
    data, labels = utils.load('./data/' + filename)

    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42, stratify=labels
    )

    run_svm_experiment(X_train, y_train, X_test, y_test, filename)