import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from enum import Enum
import logging
import random
import networkx as nx
import matplotlib.pyplot as plt
import time

from sklearn.cluster import DBSCAN
from sklearn import metrics


# from sklearn.metrics import adjusted_rand_score
# from sklearn.metrics import adjusted_mutual_info_score

def data_load(dataset_name):
    if dataset_name == 'iris':
        data = pd.read_csv('data/iris.data', header=None)
        # print(data.head())
        D_shape = data.shape
        labelCol_idx = 4
        listof_attributes = range(0, D_shape[1] - 1)

        eps_range = np.arange(start=0.5, stop=4, step=0.1)

    elif dataset_name == 'shuttle':
        data = pd.read_csv('data/iris.data', header=None)
        D_shape = data.shape
        labelCol_idx = 9
        listof_attributes = range(0, D_shape[1] - 1)

    labels_true = data.iloc[:, labelCol_idx].values
    x = data.iloc[:, listof_attributes].values

    return data, x, labels_true, eps_range, listof_attributes


def main():
    print('Sklearn dbscan for large data')

    dataset_name = 'iris'

    data, x, labels_true, eps_range, listof_attributes = data_load(dataset_name)

    minpts = 10

    plot_flag = False

    exec_time_db = np.zeros(len(eps_range))
    n_clusters_db = np.zeros(len(eps_range))
    n_noise_db = np.zeros(len(eps_range))
    arand_db = np.zeros(len(eps_range))  # Adjusted Rand Index
    amis_db = np.zeros(len(eps_range))  # Adjusted Mutual Information Score

    for i in range(len(eps_range)):
        eps = eps_range[i]

        # print('epsilon = '+str(eps))
        start_time = time.time()

        # DBSCAN algorithm from sklearn
        db = DBSCAN(eps, minpts, ).fit(x)

        endtime = time.time()
        exec_time_db[i] = endtime - start_time
        # print("---DBSCAN exec time =  %s seconds ---" % (exec_time_db[i]))

        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels_db = db.labels_

        # ref : https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py
        # Number of clusters in labels_db, ignoring noise if present.
        n_clusters_db[i] = len(set(labels_db)) - (1 if -1 in labels_db else 0)
        n_noise_db[i] = list(labels_db).count(-1)

        # print('Estimated number of clusters: %d' % n_clusters_db[i])
        # print('Estimated number of noise points: %d' % n_noise_db[i])
        # print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels_db))
        # print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels_db))
        # print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels_db))

        arand_db[i] = metrics.adjusted_rand_score(labels_true, labels_db)
        # print("Adjusted Rand Index: %0.3f" % arand_db[i])
        amis_db[i] = metrics.adjusted_mutual_info_score(labels_true, labels_db)
        # print("Adjusted Mutual Information: %0.3f"% amis_db[i])
        # print("Silhouette Coefficient: %0.3f"
        #       % metrics.silhouette_score(x, labels_db))

        # #############################################################################
        # Plot result
        if plot_flag:
            import matplotlib.pyplot as plt

            # Black removed and is used for noise instead.
            unique_labels = set(labels_db)
            colors = [plt.cm.Spectral(each)
                      for each in np.linspace(0, 1, len(unique_labels))]
            for k, col in zip(unique_labels, colors):
                if k == -1:
                    # Black used for noise.
                    col = [0, 0, 0, 1]

                class_member_mask = (labels_db == k)

                xy = x[class_member_mask & core_samples_mask]
                plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                         markeredgecolor='k', markersize=14)

                xy = x[class_member_mask & ~core_samples_mask]
                plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                         markeredgecolor='k', markersize=6)

            plt.title('Estimated number of clusters: %d' % n_clusters_db)
            plt.show()


if __name__ == '__main__':
    main()
