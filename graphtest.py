import numpy as np
import pandas as pd
import networkx as nx
from annoy import AnnoyIndex


def main():
    G = nx.Graph()
    G.add_nodes_from([2, 3, 4, 5, 6])
    G.add_edge(2, 3)
    print(G.nodes())
    print(nx.connected_components(G))
    for i in nx.connected_components(G):
        print(type(i))
        print(i)

    data = pd.read_csv('data/ann/shuttle-unsupervised-trn.csv', header=None)
    t = AnnoyIndex(9, 'euclidean')
    for i in data.index:
        t.add_item(i, data.iloc[i, 0:9])

    t.build(10)
    res = t.get_nns_by_item(2, 6, include_distances=False)
    res2 = t.get_nns_by_item(2, 6, include_distances=True)
    a, b = res2 = t.get_nns_by_item(2, 6, include_distances=True)
    print(res)
    print(res2)
    res2[1].remove(res2[0].index(2))
    res2[0].remove(2)
    print(res2)
    eps = 1.8
    mapelementinradius = list(map(lambda x: x >= eps, res2[1]))
    aaaa = np.array(list(map(lambda x: x >= eps, res2[1])))
    print(aaaa)
    print(mapelementinradius)
    elementindeicesinradius = (np.array(res2[0]))[mapelementinradius]
    print(elementindeicesinradius)


if __name__ == '__main__':
    main()
