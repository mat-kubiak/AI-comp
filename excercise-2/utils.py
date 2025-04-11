import csv
import numpy as np

files = [
    '2_1.csv',
    '2_2.csv',
    '2_3.csv',
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

