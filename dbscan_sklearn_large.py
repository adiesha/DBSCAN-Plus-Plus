import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from enum import Enum
import logging
import random
import networkx as nx
import matplotlib.pyplot as plt
import time
import dbscanpp as dbp


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

        m = 3
        n = D_shape[0]


        eps_range = np.arange(start=0.5, stop=4, step=0.1)

    elif dataset_name == 'shuttle':
        data = pd.read_csv('data/iris.data', header=None)
        D_shape = data.shape
        labelCol_idx = 9
        listof_attributes = range(0, D_shape[1] - 1)

    labels_true = data.iloc[:, labelCol_idx].values
    x = data.iloc[:, listof_attributes].values

    factor = m / n
    return data, x, labels_true, eps_range, listof_attributes, m, n, factor, labelCol_idx


def plot_clusters(x, labels_true, labels_pred, n_clusters, core_samples_mask, plot_flag):
    if plot_flag:
        import matplotlib.pyplot as plt

        # Black removed and is used for noise instead.
        unique_labels = set(labels_pred)
        colors = [plt.cm.Spectral(each)
                  for each in np.linspace(-2, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -3:
                # Black used for noise.
                col = [-2, 0, 0, 1]

            class_member_mask = (labels_pred == k)

            xy = x[class_member_mask & core_samples_mask]
            plt.plot(xy[:, -2], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=12)

            xy = x[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:, -2], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=4)

        plt.title('Estimated number of clusters: %d' % n_clusters)
        plt.show()


def main():
    print('Sklearn dbscan for large data')

    dataset_name = 'iris'

    data, x, labels_true, eps_range, listof_attributes, m, n, factor, labelCol_idx = data_load(dataset_name)

    minpts = 10

    plot_flag = False

    exec_time_db = np.zeros(len(eps_range))
    n_clusters_db = np.zeros(len(eps_range))
    n_noise_db = np.zeros(len(eps_range))
    arand_db = np.zeros(len(eps_range))  # Adjusted Rand Index
    amis_db = np.zeros(len(eps_range))  # Adjusted Mutual Information Score

    exec_time_dbp_uni = np.zeros(len(eps_range))
    n_clusters_dbp_uni = np.zeros(len(eps_range))
    n_noise_dbp_uni = np.zeros(len(eps_range))
    arand_dbp_uni = np.zeros(len(eps_range))  # Adjusted Rand Index
    amis_dbp_uni = np.zeros(len(eps_range))  # Adjusted Mutual Information Score

    exec_time_dbp_kg = np.zeros(len(eps_range))
    n_clusters_dbp_kg = np.zeros(len(eps_range))
    n_noise_dbp_kg = np.zeros(len(eps_range))
    arand_dbp_kg = np.zeros(len(eps_range))  # Adjusted Rand Index
    amis_dbp_kg = np.zeros(len(eps_range))  # Adjusted Mutual Information Score

    for i in range(len(eps_range)):
        eps = eps_range[i]
        # print('epsilon = '+str(eps))

        start_time = time.time()
        # DBSCAN algorithm from sklearn
        db = DBSCAN(eps=eps, min_samples=minpts).fit(x)
        endtime = time.time()
        exec_time_db[i] = endtime - start_time
        # print("---DBSCAN exec time =  %s seconds ---" % (exec_time_db[i]))

        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels_db = db.labels_

        # Plot clusters
        plot_clusters(x, labels_true, labels_db, n_clusters_db, core_samples_mask, plot_flag)

        # dbscan++ with uniform initialization

        result_dbp_uni, exec_time_dbp_uni[i], qc = dbp.dbscanp(data.copy(), len(listof_attributes), eps, minpts, factor,
                             initialization=dbp.Initialization.UNIFORM, plot=plot_flag)
        labels_dbp_uni = np.array(result_dbp_uni[labelCol_idx + 1])

        # dbscan++ with k greedy initialization
        result_dbp_kg, exec_time_dbp_kg[i], qc = dbp.dbscanp(data.copy(), len(listof_attributes), eps, minpts, factor,
                               initialization=dbp.Initialization.KCENTRE, plot=plot_flag)
        labels_dbp_kg = np.array(result_dbp_kg[labelCol_idx + 1])


        # ref : https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py
        # Number of clusters in labels_db, ignoring noise if present.
        n_clusters_db[i] = len(set(labels_db)) - (1 if -1 in labels_db else 0)
        n_clusters_dbp_uni[i] = len(set(labels_dbp_uni)) - (1 if -1 in labels_dbp_uni else 0)
        n_clusters_dbp_kg[i] = len(set(labels_dbp_kg)) - (1 if -1 in labels_dbp_kg else 0)

        n_noise_db[i] = list(labels_db).count(-1)
        n_noise_dbp_uni[i] = list(labels_dbp_uni).count(-1)
        n_noise_dbp_kg[i] = list(labels_dbp_kg).count(-1)


        # print('Estimated number of clusters: %d' % n_clusters_db[i])
        # print('Estimated number of noise points: %d' % n_noise_db[i])
        # print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels_db))
        # print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels_db))
        # print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels_db))

        arand_db[i] = metrics.adjusted_rand_score(labels_true, labels_db)
        arand_dbp_uni[i] = metrics.adjusted_rand_score(labels_true, labels_dbp_uni)
        arand_dbp_kg[i] = metrics.adjusted_rand_score(labels_true, labels_dbp_kg)
        # print("Adjusted Rand Index: %0.3f" % arand_db[i])
        amis_db[i] = metrics.adjusted_mutual_info_score(labels_true, labels_db)
        amis_dbp_uni[i] = metrics.adjusted_mutual_info_score(labels_true, labels_dbp_uni)
        amis_dbp_kg[i] = metrics.adjusted_mutual_info_score(labels_true, labels_dbp_kg)
        # print("Adjusted Mutual Information: %0.3f"% amis_db[i])
        # print("Silhouette Coefficient: %0.3f"
        #       % metrics.silhouette_score(x, labels_db))

    d = {'epsilon': eps_range, 'n_clusters_db': n_clusters_db, 'n_noise_db': n_noise_db,
         'ARAND_db': arand_db, 'AMIS_db': amis_db, 'Exec_time_db': exec_time_db,
         'n_clusters_dbp_uni': n_clusters_dbp_uni, 'n_noise_dbp_uni': n_noise_dbp_uni,
         'ARAND_dbp_uni': arand_dbp_uni, 'AMIS_dbp_uni': amis_dbp_uni, 'Exec_time_dbp_uni': exec_time_dbp_uni,
         'n_clusters_dbp_kg': n_clusters_dbp_kg, 'n_noise_dbp_kg': n_noise_dbp_kg,
         'ARAND_dbp_kg': arand_dbp_kg, 'AMIS_dbp_kg': amis_dbp_kg, 'Exec_time_dbp_kg': exec_time_dbp_kg}
    results = pd.DataFrame(d)
    print(results.head())
    results.to_csv('Results_large_data/{0}_results.csv'.format(dataset_name))


if __name__ == '__main__':
    main()
