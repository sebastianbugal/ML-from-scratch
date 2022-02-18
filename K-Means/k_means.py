from re import A
import distance
import numpy as np
import random

def choose_random_k(data, num_k):
    return [random.choice(data) for i in range(num_k) ]

def mean_of_centroid(data):
    return np.array(data).mean(axis=0)

def assignment(data, k):
    temp_y = []
    for i in data:
        min_distance = None
        cur_centroid = 0
        for j, val in enumerate(k):
            cur_distance =  distance.euclidean(val,  i)
            if min_distance is None: min_distance = cur_distance
            elif min_distance > cur_distance:
                min_distance = cur_distance
                cur_centroid = j
        temp_y.append(cur_centroid)
    return temp_y


def k_means(data, num_clusters = 2, initialization_method = 'rand', distance = 'euclidean'):
    k = choose_random_k(data, num_clusters)
    prev = None
    while prev != k:
        prev = k
        y = assignment(data, k)
        for i in range(len(k)):
            temp_data = []
            for j in range(len(data)):
                if y[j] == i:
                    temp_data.append(data[j])
            k[i] =  mean_of_centroid(temp_data)
    print('Converged')
    y = assignment(data, k)
    return {'y': y, 'k': k}