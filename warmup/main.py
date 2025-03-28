import csv

import numpy as np

import matplotlib.pyplot as plt

from scipy.spatial import Voronoi, voronoi_plot_2d

from sklearn import cluster
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

color_map = {
    0: '#FF8002',
    1: '#4DAD4A',
    2: '#377EB8',
}

pad_r = 1.07

def load(path):
    """
    Funkcja powinna wczytywać plik CSV, którego lokalizacja wskazywana jest przez argument
    oraz zwracać dwie tablice NumPy o rozmiarach Nxn oraz N, gdzie N to liczba obiektów,
    a n to liczba wymiarów. Tablice te odpowiadają cechom N obiektów w n-wymiarowej przestrzeni
    (liczby rzeczywiste) oraz ich etyketom (liczby całkowite od 0 do L-1 gdzie L to liczba
    etykiet). Zakładamy, że w pliku CSV jest N linii odpowiadających obiektom, a każda linia
    zaweira n+1 liczb odpowiadających wpierw kolejnym cechom obiektu (n wartości) i jego
    etykiecie (1 wartość). Liczby w każdej linii pliku CSV oddzielone są średnikami.
    """

    points = []

    with open(path, newline='') as file:
        spamreader = csv.reader(file, delimiter=';', quotechar='|')
        for row in spamreader:
            points.append(row)

    points_np = np.array(points)

    labels = points_np[:,-1]
    data = points_np[:, :-1]

    return (data, labels)

def plot_voronoi_diagram(X, y_true, y_pred):
    """
    Funkcja rysująca diagram Woronoja dla obiektów opisanych tablicą X rozmiaru Nx2 (N to liczba
    obiektów) pogrupowanych za pomocą etykiet y_pred (tablica liczby całkowitych o rozmiarze N).
    Parametr y_true może być równy None, i wtedy nie znamy prawdziwich etykiet, lub być tablicą
    N elementową z prawdziwymi etykietami. Rysując diagram należy zadbać, aby wszystkie obiekty
    były widoczne. Wszystkie rozważane tablice są tablicami NumPy.
    """

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

    for point_id, region_id in enumerate(vor.point_region):
        region = vor.regions[region_id]
        if not -1 in region and point_id < N:
            color = color_map[y_pred[point_id]]
            polygon = vor.vertices[region]
            plt.fill(*zip(*polygon), color=color, alpha=0.4)

    if y_true is not None:
        point_c = [color_map[int(float(i))] for i in y_true]
    else:
        point_c = 'black'

    plt.scatter(X[:, 0], X[:, 1], c=point_c, zorder=10)

    plt.xlim((x_min*pad_r, x_max*pad_r))
    plt.ylim((y_min*pad_r, y_max*pad_r))
    plt.savefig(f'voronoi_{'color' if y_true is not None else 'nocolor'}.png', dpi=200)
    plt.show()

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

    N = X.shape[0]

    x_max = np.max(X[:, 0])
    x_min = np.min(X[:, 0])

    y_max = np.max(X[:, 1])
    y_min = np.min(X[:, 1])

    step = 0.01

    xx = np.arange(x_min*pad_r, x_max*pad_r, step)
    yy = np.arange(y_min*pad_r, y_max*pad_r, step)

    xx, yy = np.meshgrid(xx, yy)

    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))

    grid = np.hstack((r1, r2))

    yhat = func(grid)
    yhat = np.array([float(x) for x in yhat])
    zz = yhat.reshape(xx.shape)

    levels = [0.0, 1.0, 2.0]

    colors = [color_map[int(x)] for x in levels]
    colors.append('white')

    plt.contourf(xx, yy, zz, levels=levels, colors=colors, extend='both', alpha=0.4)
    plt.contour(xx, yy, zz, levels=levels, colors='black', extend='both', linewidths=1.0)

    point_c = [color_map[int(float(i))] for i in y_true]
    plt.scatter(X[:, 0], X[:, 1], c=point_c)
    
    plt.savefig(f'boundary.png', dpi=200)
    plt.show()

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
