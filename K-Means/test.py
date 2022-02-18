from zmq import MECHANISM
from sklearn.datasets import make_blobs
from pandas import DataFrame
from matplotlib import pyplot
import k_means

if __name__ == "__main__":
    X, y = make_blobs(n_samples=200, centers=2, n_features=2)
    print(k_means.k_means(X))
    df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
    colors = {0:'red', 1:'blue', 2:'green'}
    fig, ax = pyplot.subplots()
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
    pyplot.show()
