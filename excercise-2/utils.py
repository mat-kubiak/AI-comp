import csv
import numpy as np
from matplotlib import pyplot as plt
from pyomo.contrib.parmest.graphics import sns

files = [
    '2_1.csv',
    '2_2.csv',
    '2_3.csv',
]

def load(path):
    points = []

    with open(path, newline='') as file:
        spamreader = csv.reader(file, delimiter=';', quotechar='|')
        for row in spamreader:
            points.append(row)

    points_np = np.array(points).astype(np.float32)

    labels = points_np[:,-1]
    data = points_np[:, :-1]

    return (data, labels)

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



def plot_confusion_matrix(cm, classes, title):
    # Plot confusion matrix using Seaborn heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
