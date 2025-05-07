import numpy as np
import torch
from matplotlib import pyplot as plt
from pyomo.contrib.parmest.graphics import sns
from scipy.spatial import Voronoi, voronoi_plot_2d
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def plot_decision_boundary(model, X, y, title, device):
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
    plt.savefig(f"images/boundary.png")
    plt.close()

def plot_voronoi_diagram(X, y_pred, n_clusters, y_true=None, filename=None, pad_r=1.07):
    N = X.shape[0]

    x_max = np.max(X[:, 0])
    x_min = np.min(X[:, 0])

    y_max = np.max(X[:, 1])
    y_min = np.min(X[:, 1])

    vertices = np.array([
        [x_min, y_min],
        [x_max, y_min],
        [x_min, y_max],
        [x_max, y_max],
    ])

    vor = Voronoi(np.concatenate((X, vertices * 50)))

    fig, ax = plt.subplots()
    fig = voronoi_plot_2d(
        vor,
        ax=ax,
        show_points=False,
        point_size=10,
        line_alpha=0.1,
        show_vertices=False
    )

    cmap = plt.colormaps['viridis']
    colors = cmap(np.linspace(0, 1, n_clusters))

    for point_id, region_id in enumerate(vor.point_region):
        region = vor.regions[region_id]
        if not -1 in region and point_id < N:
            polygon = vor.vertices[region]
            plt.fill(*zip(*polygon), color=colors[y_pred[point_id]], alpha=0.4)

    c_color = 'black'
    if y_true is not None:
        p_colors = cmap(np.linspace(0, 1, len(np.unique(y_true))))
        c_color = [p_colors[int(float(i))] for i in y_true]

    plt.scatter(X[:, 0], X[:, 1], c=c_color, zorder=10)

    plt.xlim((x_min*pad_r, x_max*pad_r))
    plt.ylim((y_min*pad_r, y_max*pad_r))

    fig.set_size_inches(8, 6)
    if filename == None:
        plt.savefig(f"voronoi_{'color' if y_true is not None else 'nocolor'}.png", dpi=200)
    else:
        plt.savefig(filename, dpi=200)
    plt.savefig(f"images/voronoi.png")
    plt.close()


def plot_confusion_matrix(cm, classes, title):
    # Plot confusion matrix using Seaborn heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.savefig(f"images/conf_matrix.png")
    plt.close()