import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def main():
    print('Hello')
    data_kc = pd.read_csv('data/ann/dbann.dbscan.kcenter.shuttle.result.csv')
    data_db = pd.read_csv('data/ann/dbann.dbscan.shuttle.result.csv')

    time_kc = data_kc['Time'].values
    arand_kc = data_kc['adj_rand'].values
    amis_kc = data_kc['adj_mut'].values
    eps_kc = data_kc['eps'].values

    time_db = data_db['Time'].values
    arand_db = data_db['adj_rand'].values
    amis_db = data_db['adj_mut'].values
    eps_db = data_db['eps'].values


    plt.figure()
    plt.plot(eps_kc, time_kc, 'r')  # K center
    plt.plot(eps_db, time_db, 'b')  # dbcan
    plt.title('Execution times for SHUTTLE data ')
    plt.savefig('data/ann/figures/time.jpg')

    plt.figure()
    plt.plot(eps_kc, arand_kc, 'r')  # K center
    plt.plot(eps_db, arand_db, 'b')  # dbcan
    plt.title('ARAND score for SHUTTLE data ')
    plt.savefig('data/ann/figures/arand.jpg')

    plt.figure()
    plt.plot(eps_kc, amis_kc, 'r')  # K center
    plt.plot(eps_db, amis_db, 'b')  # dbcan
    plt.title('AMIS for SHUTTLE data ')
    plt.savefig('data/ann/figures/amis.jpg')
    print()



if __name__== '__main__':
    main()