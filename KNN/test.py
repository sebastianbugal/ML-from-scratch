from cProfile import label
from hashlib import new
from pdb import post_mortem
from turtle import title
from sklearn.datasets import make_blobs
from pandas import DataFrame
from matplotlib import pyplot
import random
from k_nn import k_nn
import numpy as np

if __name__ == "__main__":
    X, y = make_blobs(n_samples=200, centers=2, n_features=2)
    random_index = random.randrange(0,len(X)-1)
    point = X[random_index]
    real_label = y[random_index]
    X = np.delete(X, random_index, axis=0)
    y = np.delete(y, random_index, axis=0)
    knn = k_nn(X,y)
    pred_class = knn.predict(point)
    print(f'Predicted Class: {pred_class}, Actual Class: {real_label}')
    df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
    colors = {0:'red', 1:'blue', 2:'yellow'}
    fig, ax = pyplot.subplots()
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', title = 'Actual', x='x', y='y', label=key, color=colors[key]) 
    pyplot.plot(point[0], point[1], marker="v", markersize=10, markeredgecolor=colors[real_label], color=colors[pred_class])
    pyplot.show()
