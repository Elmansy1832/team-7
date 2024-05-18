import os

import numpy as np


def load_data(data_dir='data'):
    X = []
    y = []
    labels = os.listdir(data_dir)
    label_map = {label: idx for idx, label in enumerate(labels)}

    for label in labels:
        label_dir = os.path.join(data_dir, label)
        for file in os.listdir(label_dir):
            if file.endswith('.npy'):
                X.append(np.load(os.path.join(label_dir, file)))
                y.append(label_map[label])

    X = np.array(X)
    y = np.array(y)
    return X, y, label_map
