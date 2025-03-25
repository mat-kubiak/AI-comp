import csv

import numpy as np

import matplotlib.pyplot as plt

from scipy.spatial import Voronoi, voronoi_plot_2d

from sklearn import cluster
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


def load(path):
    points = []

    with open(path, newline='') as file:
        spamreader = csv.reader(file, delimiter=';', quotechar='|')
        for row in spamreader:
            points.append(row)

    points_np = np.array(points)

    labels = points_np[:,-1]
    data = points_np[:, :-1]

    return (data, labels)

    """
    Funkcja powinna wczytywać plik CSV, którego lokalizacja wskazywana jest przez argument
    oraz zwracać dwie tablice NumPy o rozmiarach Nxn oraz N, gdzie N to liczba obiektów,
    a n to liczba wymiarów. Tablice te odpowiadają cechom N obiektów w n-wymiarowej przestrzeni
    (liczby rzeczywiste) oraz ich etyketom (liczby całkowite od 0 do L-1 gdzie L to liczba
    etykiet). Zakładamy, że w pliku CSV jest N linii odpowiadających obiektom, a każda linia
    zaweira n+1 liczb odpowiadających wpierw kolejnym cechom obiektu (n wartości) i jego
    etykiecie (1 wartość). Liczby w każdej linii pliku CSV oddzielone są średnikami.
    """
    pass


def plot_voronoi_diagram(X, y_true, y_pred):

    N = X.shape[0]

    color_map = {
        0: 'r',
        1: 'g',
        2: 'b',
    }

    vor = Voronoi(X)

    fig, ax = plt.subplots()
    fig = voronoi_plot_2d(
        vor,
        ax=ax,
        show_points=True,
        point_size=10,
        line_alpha=0.5,
        show_vertices=False
    )

    for point_id, region_id in enumerate(vor.point_region):
        region = vor.regions[region_id]
        color = color_map[y_pred[point_id]]

        if not -1 in region:
            polygon = vor.vertices[region]
            plt.fill(*zip(*polygon), color=color, alpha=0.5)

    plt.show()

    """
    Funkcja rysująca diagram Woronoja dla obiektów opisanych tablicą X rozmiaru Nx2 (N to liczba
    obiektów) pogrupowanych za pomocą etykiet y_pred (tablica liczby całkowitych o rozmiarze N).
    Parametr y_true może być równy None, i wtedy nie znamy prawdziwich etykiet, lub być tablicą
    N elementową z prawdziwymi etykietami. Rysując diagram należy zadbać, aby wszystkie obiekty
    były widoczne. Wszystkie rozważane tablice są tablicami NumPy.
    """
    pass


def plot_decision_boundary(X, y_true, func):
    """
    Funkcja rysująca granicę decyzyjną wyznaczaną przez funkcję klasyfikując func. Funkcja ta
    przyjmuje tablicę obiektów X o rozmiarze Nx2 (N to liczba obiektów) i zwraca tablicę liczb
    całkowitych o rozmiarze N zawierającą etykiety tych obiektów. W tym wypadku N może być
    dowolne. Argumenty X i y_true to tablice zawierające dane związane z tym samym problemem
    klasyfikacji (na przykład treningowe czy testowe). Pierwsza z nich ma rozmiar Nx2 i zawiera
    cechy N obiektów, druga zawiera N liczb całkowitych oznaczających prawdziwe etykiety tych
    obiektów. Rysując diagram należy zadbać, aby wszystkie obiekty były widoczne. Wszystkie
    rozważane tablice są tablicami NumPy.
    """
    pass


if __name__ == "__main__":
    X, y_true = load("./data.csv")

    X = StandardScaler().fit_transform(X)

    algorithm = cluster.KMeans(n_clusters=3)
    algorithm.fit(X)
    y_pred = algorithm.labels_.astype(int)
    plot_voronoi_diagram(X, y_true, y_pred)
    plot_voronoi_diagram(X, None, y_pred)

    algorithm = KNeighborsClassifier(n_neighbors=3)
    algorithm.fit(X, y_true)
    plot_decision_boundary(X, y_true, algorithm.predict)
