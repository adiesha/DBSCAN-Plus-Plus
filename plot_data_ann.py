import pandas as pd
import matplotlib.pyplot as plt


def data_load(name):
    if name == 'iris':
        data = pd.read_csv('Results_small_data/ann/iris_ann_results.csv')
    elif name =='libras':
        data = pd.read_csv('Results_small_data/ann/libras_ann_results.csv') 
    elif name =='mobile':
        data = pd.read_csv('Results_small_data/ann/mobile_ann_results.csv') 
    elif name =='seeds':
        data = pd.read_csv('Results_small_data/ann/seeds_ann_results.csv') 
    elif name =='wine':
        data = pd.read_csv('Results_small_data/ann/wine_ann_results.csv') 
    elif name =='zoo':
        data = pd.read_csv('Results_small_data/ann/zoo_ann_results.csv') 
    
    epsilon = data.iloc[:,0].values
    time_db_ann = data.iloc[:,1].values
    arand_db_ann = data.iloc[:,2].values
    amis_db_ann = data.iloc[:,3].values
    time_dbpk_ann = data.iloc[:,4].values
    arand_dbpk_ann = data.iloc[:,4].values
    amis_dbpk_ann = data.iloc[:,6].values
    time_dbpk = data.iloc[:,7].values
    arand_dbpk = data.iloc[:,8].values
    amis_dbpk = data.iloc[:,9].values
    time_db = data.iloc[:,10].values
    arand_db = data.iloc[:,11].values
    amis_db = data.iloc[:,12].values
   
    return epsilon, time_db_ann, arand_db_ann, amis_db_ann, time_dbpk_ann, arand_dbpk_ann, amis_dbpk_ann, time_dbpk, arand_dbpk, amis_dbpk, time_db, arand_db, amis_db

        
def main():
    names = 'iris', 'libras', 'mobile', 'seeds', 'wine', 'zoo'
    
    for i in range(len(names)):
        name = names[i]
        epsilon, time_db_ann, arand_db_ann, amis_db_ann, time_dbpk_ann, arand_dbpk_ann, amis_dbpk_ann, time_dbpk, arand_dbpk, amis_dbpk, time_db, arand_db, amis_db = data_load(name)
        
        plt.figure()
        plt.plot(epsilon, arand_db, 'b')            # dbscan (blue)
        plt.plot(epsilon, arand_dbpk, 'g')          # dbp k-center (green)
        plt.plot(epsilon, arand_db_ann, 'r',)       # dbscan ann (red)
        plt.plot(epsilon, arand_dbpk_ann, 'k',)     # dbp k-center ann (black)
        plt.xlabel('$\epsilon$')
        plt.title('Adj RAND Index for ' + str(name.upper()))
        plt.savefig('Results_small_data/ann/{0}_adj_rand.jpg'.format(name))
        
        plt.figure()
        plt.plot(epsilon, amis_db, 'b')            # dbscan (blue)
        plt.plot(epsilon, amis_dbpk, 'g')          # dbp k-center (green)
        plt.plot(epsilon, amis_db_ann, 'r',)       # dbscan ann (red)
        plt.plot(epsilon, amis_dbpk_ann, 'k',)     # dbp k-center ann (black)
        plt.xlabel('$\epsilon$')
        plt.title('Adj Mutual Info Score for ' + str(name.upper()))
        plt.savefig('Results_small_data/ann/{0}_adj_mut_score.jpg'.format(name))
        
        plt.figure()
        plt.plot(epsilon, time_db*1000, 'b')            # dbscan (blue)
        plt.plot(epsilon, time_dbpk*1000, 'g')          # dbp k-center (green)
        plt.plot(epsilon, time_db_ann*1000, 'r',)       # dbscan ann (red)
        plt.plot(epsilon, time_dbpk_ann*1000, 'k',)     # dbp k-center ann (black)
        plt.ylabel('Time (ms)')
        plt.xlabel('$\epsilon$')
        plt.title('Runtime for ' + str(name.upper()))
        plt.savefig('Results_small_data/ann/{0}_time.jpg'.format(name))
        
        

if __name__ == '__main__':
    main()