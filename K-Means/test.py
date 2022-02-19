from turtle import title
from sklearn.datasets import make_blobs
from pandas import DataFrame
from matplotlib import pyplot
import k_means

if __name__ == "__main__":
    X, y = make_blobs(n_samples=200, centers=3, n_features=2)
    ret_val = k_means.k_means(X, 3, 'rand')
    print(ret_val)
    df_test = DataFrame(dict(x=X[:,0], y=X[:,1], label=ret_val['y']))
    df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
    colors = {0:'red', 1:'blue', 2:'green'}
    fig, ax = pyplot.subplots()
    grouped = df.groupby('label')

    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', title = 'Actual', x='x', y='y', label=key, color=colors[key]) 
    # pyplot.show()

    fig2, ax2 = pyplot.subplots()
    grouped2 = df_test.groupby('label')
    for key, group in grouped2:
        group.plot(ax=ax2, kind='scatter',title = 'Test', x='x', y='y', label=key, color=colors[key]) 
    pyplot.show()
