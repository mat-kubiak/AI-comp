import csv

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import colormaps

from scipy.spatial import Voronoi, voronoi_plot_2d

from sklearn import cluster
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score

from pathlib import Path
Path("output").mkdir(parents=True, exist_ok=True)

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

def compute_k_means(x, n_clusters):
    algorithm = cluster.KMeans(n_clusters=n_clusters)
    algorithm.fit(x)
    y_pred = algorithm.labels_.astype(int)
    return y_pred

def compute_silhouette(x, n_clusters):
    y_pred = compute_k_means(x, n_clusters)
    return silhouette_score(x, y_pred)

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

def main():
    files = [
        '2_1.csv',
        '2_2.csv',
        '2_3.csv',
        '3_1.csv',
        '3_2.csv',
        '3_3.csv',
    ]

    clusters_range = range(2, 9)

    for filename in files:
        x, y = load('./data/' + filename)
        x = StandardScaler().fit_transform(x)

        s_scores = np.array([compute_silhouette(x, n) for n in clusters_range ])

        fig, ax = plt.subplots()
        ax.plot(clusters_range, s_scores)

        ax.grid(True, axis='x', linestyle='--', linewidth=1.5, alpha=0.8)
        ax.set_ylim(-1.0, 1.0)
        ax.set(xlabel='n clusters', ylabel='silhouette score', title=f'Silhouette score for `{filename}`')
        fig.savefig(f"output/{filename[0:-4]}_silhouette.png")
        plt.show()

        i_worst = np.argmin(s_scores)
        i_best = np.argmax(s_scores)
        n_worst = clusters_range[i_worst]
        n_best = clusters_range[i_best]
        print(f'\t{filename}:')
        print(f'worst case: n_clusters={n_worst} score: {s_scores[i_worst]}')
        print(f'best case: n_clusters={n_best} score: {s_scores[i_best]}')

        # worst case
        y_pred = compute_k_means(x, n_worst)
        plot_voronoi_diagram(x, y_pred, n_worst, f'output/{filename[0:-4]}_worst.png')

        # best case
        y_pred = compute_k_means(x, n_best)
        plot_voronoi_diagram(x, y_pred, n_best, f'output/{filename[0:-4]}_best.png')

if __name__ == '__main__':
    main()
