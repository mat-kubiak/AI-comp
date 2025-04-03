import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN

from pathlib import Path
Path("output").mkdir(parents=True, exist_ok=True)

import utils # utils.py

def compute_DBSCAN(x, eps):
    algo = DBSCAN(eps=eps)
    algo.fit(x)
    y_pred = algo.labels_.astype(int)
    n_clusters = len(np.unique(y_pred))
    return y_pred, n_clusters

def main():

    n_samples = 20
    eps_range = np.linspace(0.0, 2.0, n_samples+1)[1:]

    for filename in utils.files:
        x, y = utils.load('./data/' + filename)
        x = StandardScaler().fit_transform(x)

        # compute scores
        s_scores = []
        n_clusters = []
        for eps in eps_range:
            y_pred, num_clusters = compute_DBSCAN(x, eps)
            n_clusters.append(num_clusters)
            if num_clusters == 1:
                s_scores.append(np.nan)
            else:
                s_scores.append(silhouette_score(x, y_pred))
        s_scores = np.array(s_scores)
        n_clusters = np.array(n_clusters)

        # display scores plot
        fig, ax = plt.subplots()
        ax.plot(eps_range, s_scores)
        ax.grid(True, axis='x', linestyle='--', linewidth=1.5, alpha=0.8)
        ax.set_ylim(-1.0, 1.0)
        ax.set_xlim(np.min(eps_range), np.max(eps_range))
        ax.set(xlabel='eps', ylabel='silhouette score', title=f'Silhouette score for `{filename}`')

        # create text annotations
        for i, ncl in enumerate(n_clusters):
            ax.annotate(ncl, (eps_range[i], -0.8), textcoords="offset points", xytext=(10,0), ha='left')
        
        fig.savefig(f"output/{filename[0:-4]}_silhouette.png")
        plt.show()

        # check worst and best case
        i_worst = np.nanargmin(s_scores)
        i_best = np.nanargmax(s_scores)
        e_worst = eps_range[i_worst]
        e_best = eps_range[i_best]
        print(f'\t{filename}:')
        print(f'worst case: eps={e_worst} score: {s_scores[i_worst]}')
        print(f'best case: eps={e_best} score: {s_scores[i_best]}')

        # worst case
        y_pred, n_worst = compute_DBSCAN(x, e_worst)
        utils.plot_voronoi_diagram(x, y_pred, n_worst, f'output/{filename[0:-4]}_worst.png')

        # best case
        y_pred, n_best = compute_DBSCAN(x, e_best)
        utils.plot_voronoi_diagram(x, y_pred, n_best, f'output/{filename[0:-4]}_best.png')

if __name__ == '__main__':
    main()
