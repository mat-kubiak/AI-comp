import os
os.environ["OMP_NUM_THREADS"] = '1'

import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, homogeneity_score, completeness_score, silhouette_score



def compute_DBSCAN(x, eps):
    algo = DBSCAN(eps=eps)
    algo.fit(x)
    y_pred = algo.labels_.astype(int)
    n_clusters = len(set(y_pred))
    return y_pred, n_clusters

def compute_KMeans(x, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    y_pred = kmeans.fit_predict(x)
    return y_pred

def evaluate(true_labels, predicted_labels, x):
    ari = adjusted_rand_score(true_labels, predicted_labels)
    hom = homogeneity_score(true_labels, predicted_labels)
    comp = completeness_score(true_labels, predicted_labels)
    sil = silhouette_score(x, predicted_labels, metric='euclidean')
    return ari, hom, comp, sil

def choose_dataset():
    print("Choose a dataset:")
    print("1 - Iris")
    print("2 - Wine")
    print("3 - Breast Cancer Wisconsin")
    choice = input("Enter number (1/2/3): ")

    if choice == '1':
        data = load_iris()
        name = "Iris"
    elif choice == '2':
        data = load_wine()
        name = "Wine"
    elif choice == '3':
        data = load_breast_cancer()
        name = "Breast Cancer Wisconsin"
    else:
        print("Invalid choice. Defaulting to Iris.")
        data = load_iris()
        name = "Iris"

    return data.data, data.target, name

def main():
    X, y_true, dataset_name = choose_dataset()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"\n=== Running on {dataset_name} dataset ===")

    print("\n=== KMeans ===")
    n_clusters = len(np.unique(y_true))
    y_kmeans = compute_KMeans(X_scaled, n_clusters=n_clusters)
    ari_k, hom_k, comp_k, sil_k = evaluate(y_true, y_kmeans, X_scaled)
    print(f"ARI: {ari_k:.2f}, Homogeneity: {hom_k:.2f}, Completeness: {comp_k:.2f}, Silhouette: {sil_k:.2f}")

    print("\n=== DBSCAN ===")
    eps = float(input("Enter eps value for DBSCAN (e.g., 0.5): "))
    y_dbscan, n_clusters_db = compute_DBSCAN(X_scaled, eps=eps)
    ari_d, hom_d, comp_d, sil_d = evaluate(y_true, y_dbscan, X_scaled)
    print(f"ARI: {ari_d:.2f}, Homogeneity: {hom_d:.2f}, Completeness: {comp_d:.2f}, Silhouette: {sil_d:.2f}, Clusters: {n_clusters_db}")

if __name__ == "__main__":
    main()
