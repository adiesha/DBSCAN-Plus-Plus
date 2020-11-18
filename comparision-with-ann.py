import numpy as np
import dbscanann as dbann
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import adjusted_mutual_info_score
import csv


def dbscan_on_ann():
    epss = [27.5, 28, 28.5, 30, 30.3, 30.5, 30.7, 31, 31.3, 31.8, 32]
    for e in epss:
        minp = 10
        data = pd.read_csv('data/ann/shuttle-unsupervised-trn.csv', header=None)
        g_t = data[9].to_numpy()
        result = dbann.dbscanann(data, 9, eps=e, minpts=minp)
        # result[0].to_csv('data/iris.data.dbscan.result.csv', index=False, header=False)
        print("Time: ", result[1])
        print("query count", result[2])
        g_y = data[10].to_numpy()
        print(g_t)
        print(g_y)
        f_micro = f1_score(g_t, g_y, average='micro')
        f_macro = f1_score(g_t, g_y, average='macro')
        f_weighted = f1_score(g_t, g_y, average='weighted')
        adj_rand = adjusted_rand_score(g_t, g_y)
        adj_mut = adjusted_mutual_info_score(g_t, g_y)
        print(f_micro)
        print(f_macro)
        print(f_weighted)
        print(adj_rand)
        print(adj_mut)
        rr = [result[1], f_micro, f_macro, f_weighted, adj_rand, adj_mut, e, minp, result[2]]
        with open('data/ann/dbann.dbscan.shuttle.result.csv', 'a') as fd:
            writer = csv.writer(fd)
            writer.writerow(rr)

    # Time: 29.01366639137268 # 0.7774942528735632 # 0.12024204175465708 # 0.746005272085326 # 0.6785726305879781 # 0.6251698954028455 eps: 8
    # Time: 29.33179259300232 # 0.7777931034482759 # 0.1202576595589344 # 0.7461032384427129 # 0.6791888002542681 # 0.6279639266996143 eps: 8.5
    # Time: 28.953734874725342 # 0.777816091954023 # 0.11886276769152943 # 0.7455947182386548 # 0.6800339598336493 # 0.6316418191594411 eps: 9
    # Time: 29.13238286972046 # 0.7780919540229885 # 0.1188715160710272 # 0.7456495945104543 # 0.680223640483755 # 0.633502262254999 eps: 9.5
    # Time: 29.3204243183136 # 0.778206896551724 # 0.1188723768856926 # 0.7456549941732787 # 0.68018759070349 # 0.6343466036498846 eps: 10
    # Time: 29.7196946144104 # 0.7784597701149424 # 0.11889096424458614 # 0.7457715877617185 # 0.6808738431274789 # 0.6357403295886567 eps: 10.5
    # Time: 29.49364995956421 # 0.7785517241379311 # 0.11889832888639235 # 0.7458177842127945 # 0.6811788918822784 # 0.6366449043966491 eps: 11
    # Time: 28.881269931793213 # 0.7786436781609195 # 0.11890402302885629 # 0.7458535020631228 # 0.6813673379676867 # 0.6369383294983424 eps: 11.5
    # Time: 29.679087162017822 # 0.7786896551724138 # 0.11890770462108767 # 0.7458765957178958 # 0.6815409572201284 # 0.6376074837115048 eps: 12
    # Time: 28.369586944580078 # 0.7789655172413793 # 0.11899033211220483 # 0.7459633726608375 # 0.6818866960862199 # 0.6374069039711335 eps: 12.5
    # Time: 28.004361867904663 # 0.779448275862069 # 0.12036374970033757 # 0.7459977799770608 # 0.6816744387212472 # 0.6344330714565309 eps = 13
    # Time: 30.074697971343994 # 0.7792873563218391 # 0.11335790050962401 # 0.711064141716277 # 0.36100442192013465 # 0.4702269704803294 eps = 15
    # Time: 28.95717191696167  # 0.7806666666666666 # 0.11487839205498225 # 0.7110181845767819 # 0.3552322102512698 # 0.4603902514798784 eps = 20
    # Time: 28.049800157546997 # 0.7811264367816093 # 0.12218809940949865 # 0.7112082310729113 # 0.3552829032447543 # 0.45665781798914984 eps = 25
    # Time: 28.443565607070923 # 0.7807356321839081 # 0.11382867596911056 # 0.7044389202622665 # 0.261838898715228 # 0.3156788138527624 eps = 27


def dbscanp_on_ann_uniform():
    epss = [2, 2.3, 2.5, 2.8, 3, 3.3, 3.5, 3.8, 4, 4.5, 4.3, 4.8, 5, 5.3, 5.5, 5.8, 6, 6.3, 6.5, 6.8, 7, 7.5, 8, 9,
            10, 11, 12, 13, 15, 20, 25, 27, 27.5, 28, 28.5, 30, 30.3, 30.5, 30.7, 31, 31.3, 31.8, 32]
    for e in epss:
        minp = 10
        data = pd.read_csv('data/ann/shuttle-unsupervised-trn.csv', header=None)
        g_t = data[9].to_numpy()
        result = dbann.dbscanann(data, 9, eps=e, minpts=minp, factor=0.1, initialization=dbann.Initialization.UNIFORM,
                                 plot=False)
        # result[0].to_csv('data/ann/shuttle.data.dbscanp.ann.result.csv', index=False, header=False)
        print("Time: ", result[1])
        print("query count", result[2])
        g_y = data[10].to_numpy()
        print(g_t)
        print(g_y)
        f_micro = f1_score(g_t, g_y, average='micro')
        f_macro = f1_score(g_t, g_y, average='macro')
        f_weighted = f1_score(g_t, g_y, average='weighted')
        adj_rand = adjusted_rand_score(g_t, g_y)
        adj_mut = adjusted_mutual_info_score(g_t, g_y)
        print(f_micro)
        print(f_macro)
        print(f_weighted)
        print(adj_rand)
        print(adj_mut)
        rr = [result[1], f_micro, f_macro, f_weighted, adj_rand, adj_mut, e, minp, result[2]]
        with open('data/ann/dbann.dbscan.uniform.shuttle.result.csv', 'a') as fd:
            writer = csv.writer(fd)
            writer.writerow(rr)


def dbscanp_on_ann_kcenter():
    epss = [2, 2.3, 2.5, 2.8, 3, 3.3, 3.5, 3.8, 4, 4.5, 4.3, 4.8, 5, 5.3, 5.5, 5.8, 6, 6.3, 6.5, 6.8, 7, 7.5, 8, 9,
            10, 11, 12, 13, 15, 20, 25, 27, 27.5, 28, 28.5, 30, 30.3, 30.5, 30.7, 31, 31.3, 31.8, 32]
    for e in epss:
        minp = 10
        data = pd.read_csv('data/ann/shuttle-unsupervised-trn.csv', header=None)
        g_t = data[9].to_numpy()
        result = dbann.dbscanann(data, 9, eps=e, minpts=minp, factor=0.1, initialization=dbann.Initialization.KCENTRE,
                                 plot=False)
        # result[0].to_csv('data/ann/shuttle.data.dbscanp.ann.result.csv', index=False, header=False)
        print("Time: ", result[1])
        print("query count", result[2])
        g_y = data[10].to_numpy()
        print(g_t)
        print(g_y)
        f_micro = f1_score(g_t, g_y, average='micro')
        f_macro = f1_score(g_t, g_y, average='macro')
        f_weighted = f1_score(g_t, g_y, average='weighted')
        adj_rand = adjusted_rand_score(g_t, g_y)
        adj_mut = adjusted_mutual_info_score(g_t, g_y)
        print(f_micro)
        print(f_macro)
        print(f_weighted)
        print(adj_rand)
        print(adj_mut)
        print("eps: ", e)
        rr = [result[1], f_micro, f_macro, f_weighted, adj_rand, adj_mut, e, minp, result[2]]
        with open('data/ann/dbann.dbscan.kcenter.shuttle.result.csv', 'a') as fd:
            writer = csv.writer(fd)
            writer.writerow(rr)


def main():
    # dbscanp_on_ann_uniform()
    dbscanp_on_ann_kcenter()


if __name__ == '__main__':
    main()
