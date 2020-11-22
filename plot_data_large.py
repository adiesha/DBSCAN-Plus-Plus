import pandas as pd
import matplotlib.pyplot as plt


def data_load(name):
    data = pd.DataFrame()
    if name == 'iris':
        data = pd.read_csv('Results_small_data/iris_results.csv')
    elif name == 'letters':
        data = pd.read_csv('Results_large_data/letters_results.csv')
    data.head()
    epsilon = data.iloc[:,1]
    adj_rand_db = data.iloc[:, 4].values
    adj_rand_uni = data.iloc[:, 9].values
    adj_rand_kc = data.iloc[:, 14].values
    adj_mut_db = data.iloc[:, 5].values
    adj_mut_uni = data.iloc[:, 10].values
    adj_mut_kc = data.iloc[:, 13].values
    return epsilon, adj_rand_db, adj_rand_uni, adj_rand_kc, adj_mut_db, adj_mut_uni, adj_mut_kc


def main():
    names = 'iris','letters', 'fashion'
    for i in range(len(names)):
        name = names[i]
        epsilon, adj_rand_db, adj_rand_uni, adj_rand_kc, adj_mut_db, adj_mut_uni, adj_mut_kc = data_load(name)

        plt.figure()
        plt.plot(epsilon, adj_rand_db, 'b')  # dbscan (blue)
        plt.plot(epsilon, adj_rand_uni, 'g')  # uniform (green)
        plt.plot(epsilon, adj_rand_kc, 'r', )  # k-center (red)
        plt.title('Adj RAND Index for ' + str(name.upper()))
        plt.savefig('Results_large_data/{0}_adj_rand.jpg'.format(name))

        plt.figure()
        plt.plot(epsilon, adj_mut_db, 'b')  # dbscan (blue)
        plt.plot(epsilon, adj_mut_uni, 'g')  # uniform (green)
        plt.plot(epsilon, adj_mut_kc, 'r')  # k-center (red)
        plt.title('Adj Mutual Info Score for ' + str(name.upper()))
        plt.savefig('Results_large_data/{0}_adj_mutual_score.jpg'.format(name))


if __name__ == '__main__':
    main()