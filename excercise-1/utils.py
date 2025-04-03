import csv

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import colormaps

from scipy.spatial import Voronoi, voronoi_plot_2d

files = [
    '2_1.csv',
    '2_2.csv',
    '2_3.csv',
    '3_1.csv',
    '3_2.csv',
    '3_3.csv',
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

def plot_voronoi_diagram(X, y_pred, n_clusters, filename=None, pad_r=1.07):
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

    plt.scatter(X[:, 0], X[:, 1], c='black', zorder=10)

    plt.xlim((x_min*pad_r, x_max*pad_r))
    plt.ylim((y_min*pad_r, y_max*pad_r))

    if filename == None:
        plt.savefig(f'voronoi_{'color' if y_true is not None else 'nocolor'}.png', dpi=200)
    else:
        plt.savefig(filename, dpi=200)
    plt.show()