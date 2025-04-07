import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    adjusted_rand_score,
    homogeneity_score,
    completeness_score
)

OUT_DIR = 'page4'

from pathlib import Path
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

import utils # utils.py

def compute_DBSCAN(x, eps):
    algo = DBSCAN(eps=eps)
    algo.fit(x)
    y_pred = algo.labels_.astype(int)
    n_clusters = len(np.unique(y_pred))
    return y_pred, n_clusters

def main():

    n_samples = 20
    eps_range = np.linspace(0.0, 2.0, n_samples+1, endpoint=False)[1:]

    for filename in utils.files:
        x, y = utils.load('./data/' + filename)
        x = StandardScaler().fit_transform(x)

        # compute scores
        rand_scores = []
        homo_scores = []
        comp_scores = []
        n_clusters = []

        for eps in eps_range:
            y_pred, num_clusters = compute_DBSCAN(x, eps)
            rand_scores.append(adjusted_rand_score(y, y_pred))
            homo_scores.append(homogeneity_score(y, y_pred))
            comp_scores.append(completeness_score(y, y_pred))
            n_clusters.append(num_clusters)

        rand_scores = np.array(rand_scores)
        homo_scores = np.array(homo_scores)
        comp_scores = np.array(comp_scores)
        n_clusters = np.array(n_clusters)

        # display scores plot
        fig, ax = plt.subplots()
        ax.plot(eps_range, rand_scores, label="Adjusted Rand")
        ax.plot(eps_range, homo_scores, label="Homogeneity")
        ax.plot(eps_range, comp_scores, label="Completeness")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0.0, 2.0)

        ax.set(xlabel='eps')

        ax.vlines(eps_range, -1.0, 1.0, colors='black', linestyles='--', alpha=0.2)

        # create text annotations
        for i, ncl in enumerate(n_clusters):
            ax.annotate(ncl, (eps_range[i], 0.05), textcoords="offset points", xytext=(5,0), ha='left')

        fig.set_size_inches(10, 6)
        fig.savefig(f"{OUT_DIR}/{filename[0:-4]}_scores.png")
        plt.show()

        # weighted average with 100% ARI and completeness and 20% homogeneity
        combined_scores = (np.array(rand_scores) + 0.2 * np.array(homo_scores) + np.array(comp_scores)) / 2.2

        # check worst and best case
        i_worst = np.argmin(combined_scores)
        i_best = np.argmax(combined_scores)
        e_worst = eps_range[i_worst]
        e_best = eps_range[i_best]
        print(f'\t{filename}:')
        print(f'worst case: eps={e_worst}, n_clusters={n_clusters[i_worst]}, ARI: {rand_scores[i_worst]:.4f}, Homo: {homo_scores[i_worst]:.4f}, Comp: {comp_scores[i_worst]:.4f}, combined: {combined_scores[i_worst]:.4f}')
        print(f'best case: eps={e_best}, n_clusters={n_clusters[i_best]}, ARI: {rand_scores[i_best]:.4f}, Homo: {homo_scores[i_best]:.4f}, Comp: {comp_scores[i_best]:.4f}, combined: {combined_scores[i_best]:.4f}')

        # worst case
        y_pred, n_worst = compute_DBSCAN(x, e_worst)
        utils.plot_voronoi_diagram(x, y_pred, n_worst, f'{OUT_DIR}/{filename[0:-4]}_worst.png')

        # best case
        y_pred, n_best = compute_DBSCAN(x, e_best)
        utils.plot_voronoi_diagram(x, y_pred, n_best, f'{OUT_DIR}/{filename[0:-4]}_best.png')

if __name__ == '__main__':
    main()
