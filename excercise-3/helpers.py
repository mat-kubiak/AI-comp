import numpy as np
import torch
from matplotlib import pyplot as plt
from pyomo.contrib.parmest.graphics import sns
from scipy.spatial import Voronoi, voronoi_plot_2d
import os
from pathlib import Path

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
    plt.savefig(f"images/{title}y.png")
    plt.close()

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

def plot_voronoi_diagram(X, y_pred, n_clusters, y_true=None, filename=None, pad_r=1.07, title="Test"):
    N = X.shape[0]

    x_max = np.max(X[:, 0])
    x_min = np.min(X[:, 0])
    y_max = np.max(X[:, 1])
    y_min = np.min(X[:, 1])

    # Add bounding box to ensure all points are surrounded
    vertices = np.array([
        [x_min, y_min],
        [x_max, y_min],
        [x_min, y_max],
        [x_max, y_max],
    ])
    vor = Voronoi(np.concatenate((X, vertices * 50)))

    fig, ax = plt.subplots()
    voronoi_plot_2d(
        vor, ax=ax,
        show_points=False,
        point_size=10,
        line_alpha=0.1,
        show_vertices=False
    )

    cmap = plt.colormaps['viridis']

    if y_true is not None:
        p_colors = cmap(np.linspace(0, 1, len(np.unique(y_true))))
        c_color = [p_colors[int(float(i))] for i in y_true]
        labels_to_plot = y_true
    else:
        p_colors = cmap(np.linspace(0, 1, n_clusters))
        c_color = [p_colors[int(float(i))] for i in y_pred]
        labels_to_plot = y_pred

    # Fill each Voronoi cell with color based on prediction
    for point_id, region_id in enumerate(vor.point_region):
        region = vor.regions[region_id]
        if not -1 in region and point_id < N:
            polygon = vor.vertices[region]
            plt.fill(*zip(*polygon), color=p_colors[y_pred[point_id]], alpha=0.4)

    # Plot original points with class colors
    plt.scatter(X[:, 0], X[:, 1], c=c_color, edgecolors='k', zorder=10)

    # Create legend
    labels = np.unique(labels_to_plot)
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=p_colors[int(cls)], markeredgecolor='k',
                   markersize=6, label=f'Class {int(cls)}')
        for cls in labels
    ]
    plt.legend(handles=legend_elements, title="Class Labels")

    # Set plot limits
    plt.xlim((x_min * pad_r, x_max * pad_r))
    plt.ylim((y_min * pad_r, y_max * pad_r))
    fig.set_size_inches(8, 6)

    # Save and close
    if filename is None:
        filename = f"images/{title}.png"
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_confusion_matrix(cm, classes, title):
    Path('images').mkdir(parents=True, exist_ok=True)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=10, yticklabels=10)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.savefig(f"images/{title}.png")
    plt.close()
