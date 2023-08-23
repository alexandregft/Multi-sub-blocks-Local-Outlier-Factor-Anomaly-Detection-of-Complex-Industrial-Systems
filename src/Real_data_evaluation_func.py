from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import random

from sklearn.neighbors import LocalOutlierFactor

def generate_result_validation_real_data(list_corsets,X_test,Y_test,list_features_group,n_neighbors,contamination):
    
    list_lof_classifier=[]
    for i in range(len(list_features_group)):
        clf = LocalOutlierFactor(n_neighbors=n_neighbors,contamination=contamination,n_jobs=-1)
        clf.novelty=True
        # Generate coreset
        coreset_data = list_corsets[i]
        clf.fit(coreset_data)
        list_lof_classifier.append(clf)
    fdr=0

    y_pred_tot=np.zeros(len(X_test))
    for i in range(len(list_features_group)):
        clf=list_lof_classifier[i]
        x_val_transform=X_test[:,list_features_group[i]]
        y_pred=clf.predict(x_val_transform)
        y_score=-clf.score_samples(x_val_transform)
        y_pred[y_pred==1]=0
        y_pred[y_pred==-1]=1
        q = 1-contamination
        threshold = np.quantile(y_score, q)
        # Classify samples based on threshold
        y_pred = np.zeros_like(y_pred_tot)
        y_pred[y_score > threshold] = 1
        y_pred_tot=y_pred_tot+y_pred
        y_pred_tot=y_pred_tot>0

    y_true=Y_test
    
    y_pred_tot=y_pred_tot>0

    f1 = f1_score(y_true, y_pred_tot, average='macro')
    
    
    n_class_1 = np.sum(Y_test == 1)

    # Count the number of instances of class 1 that were correctly predicted
    n_class_1_retrieved = np.sum((Y_test == 1) & (y_pred_tot == 1))

    # Compute the ratio of class 1 retrieved
    ratio_class_1_retrieved = n_class_1_retrieved / n_class_1

    return(f1+ratio_class_1_retrieved)

def generate_result_validation_real_data_FA(list_corsets,X_val,X_test,Y_test,list_features_group,n_neighbors,contamination):
    random.seed(42)

    list_lof_classifier=[]
    list_val_data=[]
    for i in range(len(list_features_group)):
        clf = LocalOutlierFactor(n_neighbors=n_neighbors,contamination=contamination,n_jobs=-1)
        clf.novelty=True
        # Generate coreset
        coreset_data = list_corsets[i]
        clf.fit(coreset_data)
        list_lof_classifier.append(clf)
        
        list_threshold=[]
        
        
    y_pred_tot=np.zeros(len(X_val))
    for i in range(len(list_features_group)):
        clf=list_lof_classifier[i]
        x_val=X_val
        x_val_transform=x_val[:,list_features_group[i]]
        y_pred=clf.predict(x_val_transform)
        y_score=-clf.score_samples(x_val_transform)
        y_pred[y_pred==1]=0
        y_pred[y_pred==-1]=1
        q = 1-contamination
        threshold = np.quantile(y_score, q)
        # Classify samples based on threshold
        y_pred = np.zeros_like(y_pred_tot)
        y_pred[y_score > threshold] = 1
        y_pred_tot=y_pred_tot+y_pred
        y_pred_tot=y_pred_tot>0
        list_threshold.append(threshold)
    y_pred_tot_val=y_pred_tot
        
    y_pred_tot=np.zeros(len(X_test))
    y_table_res_all=np.zeros((len(list_features_group),len(X_test)))
    for i in range(len(list_features_group)):
        clf=list_lof_classifier[i]
        x_val_transform=X_test[:,list_features_group[i]]
        y_pred=clf.predict(x_val_transform)
        y_score=-clf.score_samples(x_val_transform)
        y_pred[y_pred==1]=0
        y_pred[y_pred==-1]=1
        q = 1-contamination
        
        
        threshold = list_threshold[i]
        # Classify samples based on threshold
        y_pred = np.zeros_like(y_pred_tot)
        y_pred[y_score > threshold] = 1
        y_pred_tot=y_pred_tot+y_pred
        y_pred_tot=y_pred_tot>0
        y_table_res_all[i,:]=y_pred
        
    y_true=Y_test
   

    return(y_pred_tot_val,y_pred_tot,y_true)




def generate_evaluation_test_real_data(list_corsets,X_val,X_test,Y_test,list_features_group,n_neighbors,contamination):

    random.seed(42)

    list_lof_classifier=[]
    list_val_data=[]
    for i in range(len(list_features_group)):
        clf = LocalOutlierFactor(n_neighbors=n_neighbors,contamination=contamination,n_jobs=-1)
        clf.novelty=True
        # Generate coreset
        coreset_data = list_corsets[i]
        clf.fit(coreset_data)
        list_lof_classifier.append(clf)
            
    y_pred_tot=np.zeros(len(X_val))

    list_threshold=[]
    for i in range(len(list_features_group)):
        clf=list_lof_classifier[i]
        x_val=X_val
        x_val_transform=x_val[:,list_features_group[i]]
        y_pred=clf.predict(x_val_transform)
        y_score=-clf.score_samples(x_val_transform)


        y_pred[y_pred==1]=0
        y_pred[y_pred==-1]=1
        q = 1-contamination
        threshold = np.quantile(y_score, q)
        # Classify samples based on threshold
        y_pred = np.zeros_like(y_pred_tot)
        y_pred[y_score > threshold] = 1
        y_pred_tot=y_pred_tot+y_pred
        y_pred_tot=y_pred_tot>0
        list_threshold.append(threshold)
    y_pred_tot_val=y_pred_tot
        
    y_pred_tot=np.zeros(len(X_test))
    y_table_res_all=np.zeros((len(list_features_group),len(X_test)))
    for i in range(len(list_features_group)):
        clf=list_lof_classifier[i]
        x_val_transform=X_test[:,list_features_group[i]]
        y_pred=clf.predict(x_val_transform)
        y_score=-clf.score_samples(x_val_transform)
        y_pred[y_pred==1]=0
        y_pred[y_pred==-1]=1
        q = 1-contamination
        threshold = list_threshold[i]
        
        # threshold = np.quantile(y_score, q)
        # Classify samples based on threshold
        y_pred = np.zeros_like(y_pred_tot)
        y_pred[y_score > threshold] = 1
        y_pred_tot=y_pred_tot+y_pred
        y_pred_tot=y_pred_tot>0
        y_table_res_all[i,:]=y_pred
        
    y_true=Y_test

    return(y_pred_tot,y_table_res_all)


def plot_false_alarms(y_pred, y_true, save_path=None):
    from collections import deque
    from bisect import insort, bisect_left
    from itertools import islice
    import matplotlib.pyplot as plt

    def running_median_insort(seq, window_size):
        """Contributed by Peter Otten"""
        seq = iter(seq)
        d = deque()
        s = []
        result = []
        for item in islice(seq, window_size):
            d.append(item)
            insort(s, item)
            result.append(s[len(d)//2])
        m = window_size // 2
        for item in seq:
            old = d.popleft()
            d.append(item)
            del s[bisect_left(s, old)]
            insort(s, item)
            result.append(s[m])
        return result
    
    
    def rolling_mean(array, window_size):
        """
        Compute a rolling mean of a 1D array using a specified window size.

        Args:
            array (np.ndarray): 1D array to compute rolling mean on.
            window_size (int): Size of the window for the rolling mean computation.

        Returns:
            np.ndarray: 1D array containing the rolling mean values.
        """
        return np.convolve(array, np.ones(window_size), 'valid') / window_size


    window_size = 200
    res_array_moving = rolling_mean(y_pred.astype(float), window_size)
    res_array_True_Y = rolling_mean(y_true.astype(float), window_size)

    fig = plt.gcf()
    size_x = 2000
    dep_x = 10000

    fig.set_size_inches(18.5, 10.5)
    plt.plot(res_array_moving[:], label='Detected by MLOF')
    plt.plot(res_array_True_Y[:], label="Automatically detected by the ICT equipment")
    plt.xlabel('First sample of the moving average window', fontsize=18)
    plt.ylabel('Ratio of detected sample', fontsize=16)
    plt.xticks(fontsize=14, rotation=90)
    plt.yticks(fontsize=14, rotation=90)
    plt.legend(fontsize=14)
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()

    
def save_res_real_data(Y_test,y_pred_tot,y_table_res_all,index_y_test,percentage_coreset,normalised,multiblock,Version):

    np.savetxt(f'../Results/Real_data/y_pred_tot_LOF_{percentage_coreset}_{normalised}_{multiblock}_{Version}.csv', y_pred_tot, delimiter=",")
    np.savetxt(f'../Results/Real_data/Y_test_LOF_{percentage_coreset}_{normalised}_{multiblock}_{Version}.csv', Y_test, delimiter=",")
    np.savetxt(f'../Results/Real_data/y_table_res_all_LOF_{percentage_coreset}_{normalised}_{multiblock}_{Version}.csv', y_table_res_all, delimiter=",")
    np.savetxt(f'../Results/Real_data/y_test_index_LOF_{percentage_coreset}_{normalised}_{multiblock}_{Version}.csv', index_y_test, delimiter=",")


    # Plot results


    plot_false_alarms(y_pred_tot,Y_test,save_path=f'../Results/Real_data/Analysis_{percentage_coreset}_{normalised}_{multiblock}_{Version}.png')
    
    

    
def save_res_real_data_FA(Y_test,y_pred_tot,y_table_res_all,index_y_test,percentage_coreset,normalised,multiblock,Version,n_tests_to_keep):

    np.savetxt(f'../Results/Real_data/FA_tuning/y_pred_tot_LOF_{percentage_coreset}_{normalised}_{multiblock}_{Version}_{n_tests_to_keep}.csv', y_pred_tot, delimiter=",")
    np.savetxt(f'../Results/Real_data/FA_tuning/Y_test_LOF_{percentage_coreset}_{normalised}_{multiblock}_{Version}_{n_tests_to_keep}.csv', Y_test, delimiter=",")
    np.savetxt(f'../Results/Real_data/FA_tuning/y_table_res_all_LOF_{percentage_coreset}_{normalised}_{multiblock}_{Version}_{n_tests_to_keep}.csv', y_table_res_all, delimiter=",")
    np.savetxt(f'../Results/Real_data/FA_tuning/y_test_index_LOF_{percentage_coreset}_{normalised}_{multiblock}_{Version}_{n_tests_to_keep}.csv', index_y_test, delimiter=",")


    # Plot results


    plot_false_alarms(y_pred_tot,Y_test,save_path=f'../Results/Real_data/FA_tuning/Analysis_{percentage_coreset}_{normalised}_{multiblock}_{Version}.png')
    
    
