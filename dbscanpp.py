import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from enum import Enum
import logging
import random
import networkx as nx
import matplotlib.pyplot as plt
import time


def dbscanp(data, k, eps, minpts, factor, initialization=None, plot=False, plotPath='data/result.png', norm=None,
            leafsize=10):
    start_time = time.time()

    querycount = 0
    # -1: Noise 0:Undefined >0 : cluster number
    c = 0
    # k is the value that shows us which first k number of columns contain the attribute data
    # Create a new column Add undefined label
    labelcolumn = len(data.columns)
    data[labelcolumn] = 0

    # Create the KDTree for the algorithm
    neighbourhoodtree = KDTree(data.iloc[:, 0:k].values, leafsize=leafsize)

    index_array = list(np.arange(0, data.shape[0], 1))
    sample = index_array
    if initialization == Initialization.NONE or factor == 1:
        sample = index_array
    elif initialization == Initialization.UNIFORM:
        sample = random.sample(index_array, int(factor * data.shape[0]))
        sample.sort()
    elif initialization == Initialization.KCENTRE:
        sample = kgreedyinitialization(data, k, int(factor * data.shape[0]), norm)
    core_points = []
    G = nx.Graph()
    G.add_nodes_from(index_array)
    for i in sample:

        neighbourhood = neighbourhoodtree.query_ball_point(data.iloc[i, 0:k], r=eps)
        querycount += 1
        neighbourhood.remove(i)
        # print(len(neighbourhood))
        if len(neighbourhood) >= minpts:
            core_points.append(i)

        logging.info(neighbourhood)
        # remove the i from neighbourhood list to create the seedset

        seedset = neighbourhood
        j = 0
        if len(neighbourhood) >= minpts:
            while j < len(seedset):
                q = seedset[j]
                G.add_edge(i, q)
                j = j + 1

    connected_components = nx.connected_components(G)
    # logging.info("Number of clusters %d", connected_components)
    for component in connected_components:
        size = len(component)
        if size > 1:
            c += 1
        for node in component:
            if size == 1:
                logging.info("noise point found, Index: %d", node)
                data._set_value(node, labelcolumn, -1)
            if size > 1:
                data._set_value(node, labelcolumn, c)

    logging.info("Query Count: %d", querycount)
    endtime = time.time()
    print("--- %s seconds ---" % (endtime - start_time))
    if plot:
        nx.draw(G)
        plt.savefig(plotPath)
        plt.show()

    return data, endtime - start_time, querycount


def kgreedyinitialization(data, k, m, norm=None):
    n = data.shape[0]
    distance = np.full(shape=(n), fill_value=np.inf, dtype=float)
    S = set()
    slicedData = data.iloc[:, 0:k]
    # slicedData[slicedData.shape[1]] = slicedData.index
    # indexCol = slicedData.shape[1] - 1
    data_numpy_a = slicedData.to_numpy()
    for p in range(0, m):
        index_max = np.argmax(distance)
        S.add(index_max)
        baseTuple = np.array(slicedData.iloc[index_max, 0:k])
        logging.info("p: %d", p)
        # slicedData[distCol] = slicedData.apply(lambda row: returnMin(baseTuple, np.array(row[0:k]), distance, row[indexCol]), axis=1)
        # np.apply_along_axis(returnMin, 1, data_numpy_a, baseTuple, distance, k)
        distance = vec_returnMin(data_numpy_a, baseTuple, distance, k)
    return S


def returnMin(row, basetuple, dista, k):
    tuple = np.array(row[0:k])
    index = row[-1]
    tempdistance = np.linalg.norm(basetuple - tuple, ord=None)
    temp = min(dista[index], tempdistance)
    dista[index] = temp
    return temp


def vec_returnMin(row, baseTuple, dista, k):
    tuple = np.array(row[:, 0:k])
    difftuple = tuple - baseTuple
    squaretuple = np.square(difftuple)
    distatuple = np.sqrt(squaretuple)
    distatuplerow = np.sum(squaretuple, axis=1)
    difftuple = np.fmin(distatuplerow, dista)
    return difftuple


def main():
    data = pd.read_csv('data/iris.data', header=None)
    result = dbscanp(data, 4, 0.485, 6, 1.0, initialization=Initialization.NONE, plot=True)
    result[0].to_csv('data/iris.data.dbscan.result.csv', index=False, header=False)
    print("Time: ", result[1])
    print("query count", result[2])
    aa = kgreedyinitialization(data, 4, 10)
    print(aa)


class Initialization(Enum):
    NONE = 1
    UNIFORM = 2
    KCENTRE = 3


if __name__ == "__main__":
    # import sys
    # sys.setrecursionlimit(10000)

    logging.basicConfig(level=logging.WARNING)
    logging.info("Start")
    main()
