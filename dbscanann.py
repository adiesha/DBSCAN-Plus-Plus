import logging
import random
import time
from enum import Enum

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from annoy import AnnoyIndex


def dbscanann(data, k, eps, minpts, factor=1.0, initialization=None, plot=False, plotPath='data/result.png',
              norm='euclidean', ntrees=10, n_jobs=-1):
    start_time = time.time()

    querycount = 0
    # -1: Noise 0:Undefined >0 : cluster number
    c = 0
    # k is the value that shows us which first k number of columns contain the attribute data
    # Create a new column Add undefined label
    labelcolumn = len(data.columns)
    data[labelcolumn] = 0

    # Create the KDTree for the algorithm
    neighbourhoodtree = AnnoyIndex(k, norm)

    for i in data.index:
        neighbourhoodtree.add_item(i, data.iloc[i, 0:k])

    neighbourhoodtree.build(ntrees, n_jobs)

    index_array = list(np.arange(0, data.shape[0], 1))
    sample = index_array
    if initialization == Initialization.NONE or factor == 1 or initialization is None:
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

        # neighbourhood = neighbourhoodtree.get_nns_by_item(i, int(minpts * 1.5), include_distances=True)
        # query 1.5 times the points the min points in case we miss some points
        neighbourhoodindices, neighbourhooddistnaces = neighbourhoodtree.get_nns_by_item(i, int(minpts * 2.0),
                                                                                         include_distances=True)
        querycount += 1

        # remove the i object and its distance from the queried result
        # neighbourhooddistnaces.remove(neighbourhoodindices.index(i))
        if i in neighbourhoodindices:
            del neighbourhooddistnaces[neighbourhoodindices.index(i)]
            neighbourhoodindices.remove(i)

        # run a lambda to get the boolean array that satisfies the radius condition: find the points that are in the
        # given radius
        tempmapelementinradius = list(map(lambda x: x <= eps, neighbourhooddistnaces))
        elementindicesinradius = np.array(neighbourhoodindices)[tempmapelementinradius]

        # if number of points are greater than or equal to the minpoints add the point as a core point and add edges to
        # neighbouring points
        if len(elementindicesinradius) >= minpts:
            core_points.append(i)
            logging.info(elementindicesinradius)
            seedset = elementindicesinradius
            j = 0
            while j < len(seedset):
                q = seedset[j]
                G.add_edge(i, q)  # adding neighbourhood edges
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
    result = dbscanann(data, 4, 0.485, 6, 1, initialization=Initialization.NONE, plot=True)
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
