import numpy as np
import pandas as pd
import networkx as nx


def main():
    G = nx.Graph()
    G.add_nodes_from([2, 3, 4, 5, 6])
    G.add_edge(2, 3)
    print(G.nodes())
    print(nx.connected_components(G))
    for i in nx.connected_components(G):
        print(type(i))
        print(i)


if __name__ == '__main__':
    main()
