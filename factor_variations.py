import numpy as np
import pandas as pd
import dbscanpp as dbp
import matplotlib.pyplot as plt
import time
from sklearn.cluster import DBSCAN
from sklearn import metrics


def data_load(dataset_name):
    if dataset_name == 'iris':
        data = pd.read_csv('data/iris/iris.data', header=None)
        D_shape = data.shape
        m = 3
        n = D_shape[0]
        eps_range = np.arange(start=1, stop=4, step=0.05)
        eps = 1.25

    elif dataset_name == 'libras':
        data = pd.read_csv('data/libras/movement_libras.data', header=None)
        D_shape = data.shape
        m = 84
        n = D_shape[0]
        eps_range = np.arange(start=1, stop=1.6, step=0.01)
        eps = 1.1

    elif dataset_name == 'mobile':
        data = pd.read_csv('data/mobile/train.csv')
        D_shape = data.shape
        m = 112
        n = D_shape[0]
        eps_range = np.arange(start=250, stop=800, step=7)
        eps = 325
    elif dataset_name == 'seeds':
        data = pd.read_table('data/seeds/seeds_dataset.txt', header=None)
        D_shape = data.shape
        m = 6
        n = D_shape[0]
        eps_range = np.arange(start=0.1, stop=9, step=0.1)
        eps = 1
    elif dataset_name == 'spam':
        # not same data as paper
        data = pd.read_csv('data/spam/spambase.data', header=None)
        D_shape = data.shape
        m = 793
        n = D_shape[0]
        eps_range = np.arange(start=0.5, stop=100, step=1)
        eps = 20
    elif dataset_name == 'wine':
        data = pd.read_csv('data/wine/wine.data', header=None)
        data['14'] = data.iloc[:, 0].values
        data = data.drop(0, 1)
        D_shape = data.shape
        m = 5
        n = D_shape[0]
        eps_range = np.arange(start=0.5, stop=500, step=5)
        eps = 50
    elif dataset_name == 'zoo':
        data = pd.read_csv('data/zoo/zoo.data', header=None)
        data['18'] = data.iloc[:, 0].values
        data = data.drop(0, 1)
        D_shape = data.shape
        m = 8
        n = D_shape[0]
        eps_range = np.arange(start=0.5, stop=3.5, step=0.05)
        eps = 2
    labelCol_idx = D_shape[1] - 1
    listof_attributes = range(0, labelCol_idx)
    labels_true = data.iloc[:, labelCol_idx].values
    x = data.iloc[:, listof_attributes].values

    factor = m / n

    factor_range = np.arange(start=0.1, stop= 1.005, step= 0.1)
    return data, x, labels_true, eps, listof_attributes, m, n, factor_range, labelCol_idx


def plot_clusters(x, labels_true, labels_pred, n_clusters, core_samples_mask, plot_flag):
    if plot_flag:

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


def results_out(dataset_name, data, x, labels_true, eps, listof_attributes, m, n,
                factor_range, labelCol_idx, minpts, plot_flag):
    exec_time_db = np.zeros(len(factor_range))
    n_clusters_db = np.zeros(len(factor_range))
    n_noise_db = np.zeros(len(factor_range))
    acc_db = np.zeros(len(factor_range))
    arand_db = np.zeros(len(factor_range))  # Adjusted Rand Index
    amis_db = np.zeros(len(factor_range))  # Adjusted Mutual Information Score

    exec_time_dbp_uni = np.zeros(len(factor_range))
    n_clusters_dbp_uni = np.zeros(len(factor_range))
    n_noise_dbp_uni = np.zeros(len(factor_range))
    acc_dbp_uni = np.zeros(len(factor_range))
    arand_dbp_uni = np.zeros(len(factor_range))  # Adjusted Rand Index
    amis_dbp_uni = np.zeros(len(factor_range))  # Adjusted Mutual Information Score

    exec_time_dbp_kg = np.zeros(len(factor_range))
    n_clusters_dbp_kg = np.zeros(len(factor_range))
    n_noise_dbp_kg = np.zeros(len(factor_range))
    acc_dbp_kg = np.zeros(len(factor_range))
    arand_dbp_kg = np.zeros(len(factor_range))  # Adjusted Rand Index
    amis_dbp_kg = np.zeros(len(factor_range))  # Adjusted Mutual Information Score

    for i in range(len(factor_range)):
        factor = factor_range[i]
        print('epsilon = ' + str(eps))

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
                                                               initialization=dbp.Initialization.UNIFORM,
                                                               plot=plot_flag)
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

        acc_db[i] = metrics.accuracy_score(labels_true, labels_db)
        acc_dbp_uni[i] = metrics.accuracy_score(labels_true, labels_dbp_uni)
        acc_dbp_kg[i] = metrics.accuracy_score(labels_true, labels_dbp_kg)

        arand_db[i] = metrics.adjusted_rand_score(labels_true, labels_db)
        arand_dbp_uni[i] = metrics.adjusted_rand_score(labels_true, labels_dbp_uni)
        arand_dbp_kg[i] = metrics.adjusted_rand_score(labels_true, labels_dbp_kg)

        amis_db[i] = metrics.adjusted_mutual_info_score(labels_true, labels_db)
        amis_dbp_uni[i] = metrics.adjusted_mutual_info_score(labels_true, labels_dbp_uni)
        amis_dbp_kg[i] = metrics.adjusted_mutual_info_score(labels_true, labels_dbp_kg)

    d = {'factor': factor_range, 'n_clusters_db': n_clusters_db, 'n_noise_db': n_noise_db, 'acc_db' : acc_db,
         'ARAND_db': arand_db, 'AMIS_db': amis_db, 'Exec_time_db': exec_time_db,
         'n_clusters_dbp_uni': n_clusters_dbp_uni, 'n_noise_dbp_uni': n_noise_dbp_uni, 'acc_dbp_uni' : acc_dbp_uni,
         'ARAND_dbp_uni': arand_dbp_uni, 'AMIS_dbp_uni': amis_dbp_uni, 'Exec_time_dbp_uni': exec_time_dbp_uni,
         'n_clusters_dbp_kg': n_clusters_dbp_kg, 'n_noise_dbp_kg': n_noise_dbp_kg, 'acc_dbp_kg' : acc_dbp_kg,
         'ARAND_dbp_kg': arand_dbp_kg, 'AMIS_dbp_kg': amis_dbp_kg, 'Exec_time_dbp_kg': exec_time_dbp_kg}
    results = pd.DataFrame(d)
    print(results.head())
    results.to_csv('Results_small_data_factor/{0}_results_factor.csv'.format(dataset_name), index=False)


def main():
    names = 'iris', 'libras', 'mobile', 'seeds', 'spam', 'wine', 'zoo'
    for i in range(len(names)):
        dataset_name = names[i]
        print('DBSCAN and DBSCANPP for', str(dataset_name))
        data, x, labels_true, eps, listof_attributes, m, n, factor_range, labelCol_idx = data_load(dataset_name)
        minpts = 10
        plot_flag = False
        results_out(dataset_name, data, x, labels_true, eps, listof_attributes, m, n, factor_range, labelCol_idx,
                    minpts, plot_flag)


if __name__ == '__main__':
    main()