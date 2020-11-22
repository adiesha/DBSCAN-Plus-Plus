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

    epsilon = data.iloc[:, 0].values
    noise_db = data.iloc[:, 2].values
    noise_uni = data.iloc[:, 6].values
    noise_kc = data.iloc[:, 10].values
    acc_db = data.iloc[:, 3].values
    acc_uni = data.iloc[:, 7].values
    acc_kc = data.iloc[:, 11].values
    return epsilon, noise_db, noise_uni, noise_kc, acc_db, acc_uni, acc_kc


def main():
    names = 'iris', 'libras', 'mobile', 'seeds', 'spam', 'wine', 'zoo'
    for i in range(len(names)):
        name = names[i]
        variation, noise_db, noise_uni, noise_kc, acc_db, acc_uni, acc_kc = data_load(name)

        plt.figure()
        plt.plot(variation, noise_db, 'b')  # dbscan (blue)
        plt.plot(variation, noise_uni, 'g')  # uniform (green)
        plt.plot(variation, noise_kc, 'r', )  # k-center (red)
        plt.title('Noise points for ' + str(name.upper()))
        plt.savefig('Results_small_data_factor/{0}_noise.jpg'.format(name))

        plt.figure()
        plt.plot(variation, acc_db, 'b')  # dbscan (blue)
        plt.plot(variation, acc_uni, 'g')  # uniform (green)
        plt.plot(variation, acc_kc, 'r')  # k-center (red)
        plt.title('Accuracy Score for ' + str(name.upper()))
        plt.savefig('Results_small_data_factor/{0}_accu_score.jpg'.format(name))


if __name__ == '__main__':
    main()