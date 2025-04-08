import numpy as np

import matplotlib.pyplot as plt

from sklearn import cluster
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    adjusted_rand_score,
    homogeneity_score,
    completeness_score
)

OUT_DIR = 'page3'

from pathlib import Path
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

import utils # utils.py

def compute_k_means(x, n_clusters):
    algorithm = cluster.KMeans(n_clusters=n_clusters)
    algorithm.fit(x)
    y_pred = algorithm.labels_.astype(int)
    return y_pred

def main():
    clusters_range = range(2, 10)

    for filename in utils.files:
        x, y = utils.load('./data/' + filename)
        x = StandardScaler().fit_transform(x)

        rand_scores = []
        homo_scores = []
        comp_scores = []

        for n in clusters_range:
            y_pred = compute_k_means(x, n)
            rand_scores.append(adjusted_rand_score(y, y_pred))
            homo_scores.append(homogeneity_score(y, y_pred))
            comp_scores.append(completeness_score(y, y_pred))

        rand_scores = np.array(rand_scores)
        homo_scores = np.array(homo_scores)
        comp_scores = np.array(comp_scores)

        fig, ax = plt.subplots()
        ax.plot(clusters_range, rand_scores, label="Adjusted Rand")
        ax.plot(clusters_range, homo_scores, label="Homogeneity")
        ax.plot(clusters_range, comp_scores, label="Completeness")

        ax.grid(True, axis='x', linestyle='--', linewidth=1.5, alpha=0.8)
        ax.set_ylim(-0.05, 1.05)
        ax.set(xlabel='n clusters')
        ax.legend()
        fig.set_size_inches(10, 6)
        fig.savefig(f"{OUT_DIR}/{filename[0:-4]}_scores.png")
        plt.show()

        # weighted average with 100% ARI and completeness and 20% homogeneity
        combined_scores = (np.array(rand_scores) + 0.2 * np.array(homo_scores) + np.array(comp_scores)) / 2.2

        i_worst = np.argmin(combined_scores)
        i_best = np.argmax(combined_scores)
        n_worst = clusters_range[i_worst]
        n_best = clusters_range[i_best]
        print(f'\t{filename}:')
        print(f'worst case: n_clusters={n_worst} ARI: {rand_scores[i_worst]:.4f}, Homo: {homo_scores[i_worst]:.4f}, Comp: {comp_scores[i_worst]:.4f}, combined: {combined_scores[i_worst]:.4f}')
        print(f'best case: n_clusters={n_best} ARI: {rand_scores[i_best]:.4f}, Homo: {homo_scores[i_best]:.4f}, Comp: {comp_scores[i_best]:.4f}, combined: {combined_scores[i_best]:.4f}')

        # worst case
        y_pred = compute_k_means(x, n_worst)
        utils.plot_voronoi_diagram(x, y_pred, n_worst, y_true=y, filename=f'{OUT_DIR}/{filename[0:-4]}_worst.png')

        # best case
        y_pred = compute_k_means(x, n_best)
        utils.plot_voronoi_diagram(x, y_pred, n_best, y_true=y, filename=f'{OUT_DIR}/{filename[0:-4]}_best.png')

if __name__ == '__main__':
    main()
