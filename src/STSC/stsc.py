import numpy as np

import sys
sys.path.append('../')
from src.STSC.stsc_ulti import affinity_to_lap_to_eig, reformat_result, get_min_max
from src.STSC.stsc_np import get_rotation_matrix as get_rotation_matrix_np
from src.STSC.stsc_autograd import get_rotation_matrix as get_rotation_matrix_autograd


def self_tuning_spectral_clustering(affinity, get_rotation_matrix, min_n_cluster=None, max_n_cluster=None):
    w, v = affinity_to_lap_to_eig(affinity)
    min_n_cluster, max_n_cluster = get_min_max(w, min_n_cluster, max_n_cluster)
    re = []
    list_cost=[]
    list_cluster=[]
    for c in range(min_n_cluster, max_n_cluster + 1):
        x = v[:, -c:]
        cost, r = get_rotation_matrix(x, c)
        re.append((cost, x.dot(r)))
        print('n_cluster: %d \t cost: %f' % (c, cost))
        list_cost.append(cost)
        list_cluster.append(c)
    COST, Z = sorted(re, key=lambda x: x[0])[0]
    return reformat_result(np.argmax(Z, axis=1), Z.shape[0]),list_cost,list_cluster


def self_tuning_spectral_clustering_np(affinity, min_n_cluster=None, max_n_cluster=None):
    return self_tuning_spectral_clustering(affinity, get_rotation_matrix_np, min_n_cluster, max_n_cluster)


def self_tuning_spectral_clustering_autograd(affinity, min_n_cluster=None, max_n_cluster=None):
    return self_tuning_spectral_clustering(affinity, get_rotation_matrix_autograd, min_n_cluster, max_n_cluster)


def self_tuning_spectral_clustering_manopt(affinity, min_n_cluster=None, max_n_cluster=None):
    return self_tuning_spectral_clustering(affinity, get_rotation_matrix_manopt, min_n_cluster, max_n_cluster)
