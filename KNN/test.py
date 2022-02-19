from hashlib import new
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
    X = np.delete(X, random_index, axis=0)
    knn = k_nn(X,y)
    knn.predict(point)
    # df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
    # colors = {0:'red', 1:'blue', 2:'yellow'}
    # fig, ax = pyplot.subplots()
    # grouped = df.groupby('label')

    # for key, group in grouped:
    #     group.plot(ax=ax, kind='scatter', title = 'Actual', x='x', y='y', label=key, color=colors[key]) 
    # # pyplot.show()

    # fig2, ax2 = pyplot.subplots()
    # grouped2 = df_test.groupby('label')
    # for key, group in grouped2:
    #     group.plot(ax=ax2, kind='scatter',title = 'Test', x='x', y='y', label=key, color=colors[key]) 
    # pyplot.show()
