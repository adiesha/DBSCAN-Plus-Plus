import dbscanpp
import dbscanann as dbann
import numpy as np
import pandas as pd
import for_small_datasets as smd
from sklearn import metrics


def main():
    # names = 'iris', 'libras', 'mobile', 'seeds', 'spam', 'wine', 'zoo'
    names = 'iris', 'libras'
    
    # for i in range(len(names)):
    #     if names[i] == 'iris':
    #         eps_range = np.arange(start=0.01, stop=1, step=0.05)
    #         eps_range = np.append(eps_range, np.arange(start=4, stop=10, step=0.2))
            
    #     elif names[i] == 'libras':
    #         eps_range = np.arange(start=0.01, stop=1, step=0.05)
    #         eps_range = np.append(eps_range, np.arange(start=1.6, stop=3.6, step=0.05))
    
    
    for i in range(len(names)):
        dataset_name = names[i]
        
        data, x, labels_true, eps_range, listof_attributes, m, n, factor, labelCol_idx = smd.data_load(dataset_name)
        
        if names[i] == 'iris':
            eps_range = np.arange(start=0.01, stop=1, step=0.05)
            eps_range = np.append(eps_range, np.arange(start=4, stop=10, step=0.2))
            
        elif names[i] == 'libras':
            eps_range = np.arange(start=0.01, stop=1, step=0.05)
            eps_range = np.append(eps_range, np.arange(start=1.6, stop=3.6, step=0.05))
        
        minpts = 10
        factor = 0.1
        plot_flag = False
        print('DBSCANPP with ANN for', str(names[i]))
        
        time_db_ann = np.zeros(len(eps_range))
        arand_db_ann = np.zeros(len(eps_range))
        amis_db_ann = np.zeros(len(eps_range))
        
        time_dbp_k_ann = np.zeros(len(eps_range))
        arand_dbp_k_ann = np.zeros(len(eps_range))
        amis_dbp_k_ann = np.zeros(len(eps_range))
        
        time_dbp_u_ann = np.zeros(len(eps_range))
        arand_dbp_u_ann = np.zeros(len(eps_range))
        amis_dbp_u_ann = np.zeros(len(eps_range))
        
        time_db = np.zeros(len(eps_range))
        arand_db = np.zeros(len(eps_range))
        amis_db = np.zeros(len(eps_range))
        
        for e in range(len(eps_range)):
            # DBSCAN with ANN
            result_db_ann, time_db_ann[e], q = dbann.dbscanann(data, labelCol_idx, eps=eps_range[e], minpts=minpts, factor=1, initialization=dbann.Initialization.NONE, plot=plot_flag)
            labels_db_ann = np.array(result_db_ann[labelCol_idx + 1])
            arand_db_ann[e] = metrics.adjusted_rand_score(labels_true, labels_db_ann)
            amis_db_ann[e] = metrics.adjusted_mutual_info_score(labels_true, labels_db_ann)
            
            # DBSCANPP K-Center with ANN
            result_dbp_k_ann, time_dbp_k_ann[e], q = dbann.dbscanann(data, labelCol_idx, eps=eps_range[e], minpts=minpts, factor=factor, initialization=dbann.Initialization.KCENTRE,
                                 plot=plot_flag)
            labels_dbp_k_ann = np.array(result_dbp_k_ann[labelCol_idx + 1])
            arand_dbp_k_ann[e] = metrics.adjusted_rand_score(labels_true, labels_dbp_k_ann)
            amis_dbp_k_ann[e] = metrics.adjusted_mutual_info_score(labels_true, labels_dbp_k_ann)
            
            # DBSCANPP Uniform with ANN
            result_dbp_u_ann, time_dbp_u_ann[e], q = dbann.dbscanann(data, labelCol_idx, eps=eps_range[e], minpts=minpts, factor=factor, initialization=dbann.Initialization.UNIFORM, plot=plot_flag)
            labels_dbp_u_ann = np.array(result_dbp_u_ann[labelCol_idx + 1])
            arand_dbp_u_ann[e] = metrics.adjusted_rand_score(labels_true, labels_dbp_u_ann)
            amis_dbp_u_ann[e] = metrics.adjusted_mutual_info_score(labels_true, labels_dbp_u_ann)
            
            # DBSCAN
            result_db, time_db[e], q = dbscanpp.dbscanp(data, labelCol_idx, eps=eps_range[e], minpts=minpts, factor=1, plot=plot_flag)
            labels_db = np.array(result_db[labelCol_idx + 1])
            arand_db[e] = metrics.adjusted_rand_score(labels_true, labels_db)
            amis_db[e] = metrics.adjusted_mutual_info_score(labels_true, labels_db)
            
        val = {'Epsilon': eps_range, 'TIME_DB_ANN': time_db_ann, 'ARAND_DB_ANN': 
               arand_db_ann, 'AMIS_DB_ANN': amis_db_ann, 'TIME_DBPK_ANN': 
                   time_dbp_k_ann, 'ARAND_DBPK_ANN': arand_dbp_k_ann, 
                   'AMIS_DBPK_ANN': amis_dbp_k_ann, 'TIME_DBPU_ANN': 
                       time_dbp_u_ann, 'ARAND_DBPU_ANN': arand_dbp_u_ann, 
                       'AMIS_DBPU_ANN': amis_dbp_u_ann, 'TIME_DB': time_db, 
                       'ARAND_DB': arand_db, 'AMIS_DB': amis_db}
        
        results = pd.DataFrame(val)
        results.to_csv('Results_small_data/ann/{0}_ann_results.csv'.format(dataset_name), index=False)
            
        print('Complete for', str(names[i]))
    return
        
        
        
        
        
if __name__ == '__main__':
    main()