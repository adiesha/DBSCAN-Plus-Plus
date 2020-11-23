import pandas as pd
import matplotlib.pyplot as plt


def data_load(name):
    if name == 'iris':
        data = pd.read_csv('Results_small_data_factor/iris_results_factor.csv')
    elif name == 'libras':
        data = pd.read_csv('Results_small_data_factor/libras_results_factor.csv')
    elif name == 'mobile':
        data = pd.read_csv('Results_small_data_factor/mobile_results_factor.csv')
    elif name == 'seeds':
        data = pd.read_csv('Results_small_data_factor/seeds_results_factor.csv')
    elif name == 'spam':
        data = pd.read_csv('Results_small_data_factor/spam_results_factor.csv')
    elif name == 'wine':
        data = pd.read_csv('Results_small_data_factor/wine_results_factor.csv')
    elif name == 'zoo':
        data = pd.read_csv('Results_small_data_factor/zoo_results_factor.csv')

    factor = data['factor'].values
    noise_db = data['n_noise_db'].values
    noise_uni = data['n_noise_dbp_uni'].values
    noise_kc = data['n_noise_dbp_kg'].values
    acc_db = data['acc_db'].values
    acc_uni = data['acc_dbp_uni'].values
    acc_kc = data['acc_dbp_kg'].values
    arand_db = data['ARAND_db'].values
    arand_dbp_uni = data['ARAND_dbp_uni'].values
    arand_dbp_kg = data['ARAND_dbp_kg'].values
    amis_db = data['AMIS_db'].values
    amis_dbp_uni = data['AMIS_dbp_uni'].values
    amis_dbp_kg = data['AMIS_dbp_kg'].values
    time_db = data['Exec_time_db'].values
    time_dbp_uni = data['Exec_time_dbp_uni'].values
    time_dbp_kg = data['Exec_time_dbp_kg'].values

    return factor, noise_db, noise_uni, noise_kc, acc_db, acc_uni, acc_kc, arand_db, arand_dbp_uni, arand_dbp_kg, \
           amis_db, amis_dbp_uni, amis_dbp_kg, time_db, time_dbp_uni, time_dbp_kg


def main():
    names = 'iris', 'libras', 'mobile', 'seeds', 'spam', 'wine', 'zoo'
    for i in range(len(names)):
        name = names[i]
        factor, noise_db, noise_uni, noise_kc, acc_db, acc_uni, acc_kc, arand_db, arand_dbp_uni, arand_dbp_kg, \
        amis_db, amis_dbp_uni, amis_dbp_kg, time_db, time_dbp_uni, time_dbp_kg = data_load(name)

        # plt.figure()
        # plt.plot(factor, noise_db, 'b')  # dbscan (blue)
        # plt.plot(factor, noise_uni, 'g')  # uniform (green)
        # plt.plot(factor, noise_kc, 'r', )  # k-center (red)
        # plt.title('Noise points for ' + str(name.upper()))
        # plt.savefig('Results_small_data_factor/{0}_noise.jpg'.format(name))

        # plt.figure()
        # plt.plot(factor, amis_dbp_uni, 'g:')  # dbscan (blue)
        # plt.plot(factor, arand_dbp_uni, 'g--')  # uniform (green)
        # plt.plot(factor, amis_dbp_kg, 'r:')  # dbscan (blue)
        # plt.plot(factor, arand_dbp_kg, 'r--')  # uniform (green)
        # plt.plot(factor, amis_db, 'b:')
        # plt.plot(factor, arand_db, 'b--')
        # # plt.plot(factor, time_dbp_uni, 'r')  # k-center (red)
        # plt.title('DBSCAN++ performance scores for ' + str(name.upper()))
        # plt.savefig('Results_small_data_factor/{0}_perf_score.jpg'.format(name))

        # plt.figure()
        # plt.plot(factor, amis_dbp_kg, 'b')  # dbscan (blue)
        # plt.plot(factor, arand_dbp_kg, 'g')  # uniform (green)
        # # plt.plot(factor, time_dbp_uni, 'r')  # k-center (red)
        # plt.title('DBSCAN++ K-center performance scores for ' + str(name.upper()))
        # plt.savefig('Results_small_data_factor/{0}_perf_score_kc.jpg'.format(name))

        plt.figure()
        # plt.plot(factor, time_db, 'b')  # dbscan (blue)
        plt.plot(factor, time_dbp_uni, 'g')  # uniform (green)
        plt.plot(factor, time_dbp_kg, 'r')  # k-center (red)
        plt.title('DBSCAN++  execution time for ' + str(name.upper()))
        plt.savefig('Results_small_data_factor/{0}_time.jpg'.format(name))



if __name__ == '__main__':
    main()