import numpy as np

import matplotlib.pyplot as plt

from sklearn import cluster
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score

from pathlib import Path
Path("output").mkdir(parents=True, exist_ok=True)

import utils # utils.py

def compute_k_means(x, n_clusters):
    algorithm = cluster.KMeans(n_clusters=n_clusters)
    algorithm.fit(x)
    y_pred = algorithm.labels_.astype(int)
    return y_pred

def main():
    clusters_range = range(2, 9)

    for filename in utils.files:
        x, y = utils.load('./data/' + filename)
        x = StandardScaler().fit_transform(x)

        # compute scores
        s_scores = []
        for n in clusters_range:
            y_pred = compute_k_means(x, n)
            s_scores.append(silhouette_score(x, y_pred))
        s_scores = np.array(s_scores)

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
        utils.plot_voronoi_diagram(x, y_pred, n_worst, f'output/{filename[0:-4]}_worst.png')

        # best case
        y_pred = compute_k_means(x, n_best)
        utils.plot_voronoi_diagram(x, y_pred, n_best, f'output/{filename[0:-4]}_best.png')

if __name__ == '__main__':
    main()
