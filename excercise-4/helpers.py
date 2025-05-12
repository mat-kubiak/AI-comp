import numpy as np
import torch
from matplotlib import pyplot as plt
from pyomo.contrib.parmest.graphics import sns
import os
from pathlib import Path
import torch.nn.functional as F

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def plot_decision_boundary(classifier_head, X, y, title, device):
    # Create a meshgrid over the input space
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))

    # Prepare the grid
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.tensor(grid, dtype=torch.float32).to(device)

    # Evaluate using only the classifier head
    classifier_head.eval()
    with torch.no_grad():
        preds = classifier_head(grid_tensor).argmax(dim=1).cpu().numpy()

    Z = preds.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.title(title)
    plt.savefig(f"images/{title}.png")
    plt.close()


def plot_confusion_matrix(cm, classes, title):
    Path('images').mkdir(parents=True, exist_ok=True)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=10, yticklabels=10)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.savefig(f"images/{title}.png")
    plt.close()


