
import pyreadr
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.svm import OneClassSVM
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def generate_result_validation(scaler_list,list_corsets,Validation_df,cols_feature,list_features_group,n_neighbors,contamination):
    
    
    len_sample_val=int(max(Validation_df['sample']))
    list_lof_classifier=[]
    for i in range(len(list_features_group)):
        scaler=scaler_list[i]
        clf = LocalOutlierFactor(n_neighbors=n_neighbors,contamination=contamination,n_jobs=-1)
        clf.novelty=True
        percentage = 0.1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Generate coreset
        coreset_data = list_corsets[i]
        clf.fit(coreset_data)
        list_lof_classifier.append(clf)
    fdr=0
    n_simu=Validation_df.simulationRun.nunique()
    len_sample=Validation_df['sample'].nunique()
    detected_issues=[]
    detected_list_glob=[]
    Tab_all_results=np.zeros((1,n_simu,len_sample))
    list_tab_res_i=[]
    y_pred_tot=np.zeros(len(Validation_df))
    for i in range(len(list_features_group)):
        clf=list_lof_classifier[i]
        scaler=scaler_list[i]
        x_val_transform=X_test[:,list_features_group[i]]
        x_val_transform=scaler.transform(x_val_transform)
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
        tab_res_i=np.zeros(len_sample)

  
    return(np.sum(y_pred_tot)/len(y_pred_tot))




def test_different_fault_lof(scaler_list,list_corsets,Testing_df,cols_feature,list_group_features,n_neighbors=study.best_params['n_neighbors'],contamination=study.best_params['contamination']):
    """Extract test result for each fault scenario of the Testing Dataset using multiple blocks of features
    An anomaly score based on the Local Outlier Factor is first train for each group of features.
    By comparing the local density of a sample to the local densities of its
    neighbors, one can identify samples that have a substantially lower density
    than their neighbors. These are considered outliers.
    
    Parameters
    --------------
    Training_df: Training dataframe containing several simulation of 'normal' scenario.
    
    Testing_df: Testing_df dataframe containing n_simu simulation of 'faulty' scenario.
    
    cols_feature: List of features name of the dataframe that will be used to compute the anomaly score.
    
    n_neighbors : int, default=40
        Number of neighbors to use by default for :meth:`kneighbors` queries.
        If n_neighbors is larger than the number of samples provided,
        all samples will be used.
        
   len_sample: int, default=960
   Number of observations for one simulation of a faulty scenario.
   
   n_simu : int, default=500
   Number of simulation of one faulty scenario.
   -----------------------------------
   Output
   --------------------------------
   list_tab_res_fault: 
   
   detected_issues: list of lenght number of faulty scenario.
   Each element of the list contains an array of size (n_simu,len_sample)
   The value of an element (i,j) of the array corresponds to number of features that found that for the simulation i,
   the observation j is an outlier.
    
    """
    n_simu = int(max(Testing_df['simulationRun']))
    len_sample = int(max(Testing_df['sample']))
    if n_neighbors is None:
        n_neighbors = study.best_params['n_neighbors']
    if contamination is None:
        contamination = study.best_params['contamination']

    list_tab_res_fault = []
    detected_issues = []
    Tab_all_results = np.zeros((max(Testing_df['faultNumber'].unique()), len(list_group_features), n_simu, len_sample))
    list_lof_classifier = []

    for i, group_features in enumerate(list_group_features):
        scaler = scaler_list[i]
        clf = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True,n_jobs=-1).fit(list_corsets[i])
        list_lof_classifier.append(clf)

    for fault_num in range(1, max(Testing_df['faultNumber'].unique())+1):
        x_test = Testing_df[Testing_df['faultNumber'] == fault_num][cols_feature].to_numpy()
        issues_tab = np.zeros((len(x_test)//len_sample, len_sample))
        y_pred_tot = np.zeros(len(x_test))

        list_tab_res_i = []
        for i, clf in enumerate(list_lof_classifier):
            x_test_transform = x_test[:, list_group_features[i]]
            x_test_transform = scaler_list[i].transform(x_test_transform)
            y_score = -clf.score_samples(x_test_transform)
            q = 1 - contamination
            threshold = np.quantile(y_score, q)

            y_pred = np.zeros_like(y_pred_tot)
            y_pred[y_score > threshold] = 1
            y_pred_tot += y_pred
            tab_res_i = np.zeros(len_sample)
            for element in range(len(y_pred)//len_sample):
                tab_res_i += y_pred[element*len_sample:(element+1)*len_sample]
                issues_tab[element, :] += y_pred[element*len_sample:(element+1)*len_sample]
                Tab_all_results[fault_num-1, i, element, :] = y_pred[element*len_sample:(element+1)*len_sample]
            list_tab_res_i.append(tab_res_i)
        list_tab_res_fault.append(list_tab_res_i)
        detected_issues.append(issues_tab)
    return Tab_all_results, detected_issues