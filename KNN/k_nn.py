from itertools import count
from re import A
from tokenize import group
from distance import euclidean
import numpy as np
import random
import pandas as pd

class k_nn:

    def __init__(self, X, Y):
        self.x = X
        self.y = Y
        self.number_of_groups = len(np.unique(self.y))

    def predict(self, point, number_of_neighbours = 3, distance = 'euclidean'):
        if number_of_neighbours % self.number_of_groups == 0: assert('Number of neighbours cannot be a multiple of k...')
        if number_of_neighbours == 1:
            min_distance, min_index = None, 0
            for i, val in enumerate(self.x):
                cur_distance = euclidean(val, point)
                if min_distance > cur_distance or min_distance is None:
                    min_distance = cur_distance
                    min_index = i
            return self.y[min_index]
        else:
            distances = [euclidean(i, point) for i in self.x]
            zip_dy = zip(distances, self.y)
            list_dy = list(zip_dy)
            list_dy.sort(key=lambda x: x[0])
            group_list = [i[1] for i in list_dy[:number_of_neighbours]]
            values, counts = np.unique(group_list, return_counts=True)
            point_group_index = np.argmax(counts)
            return values[point_group_index]




