#reading train data in .R format
import pyreadr
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler


# from Article_journal_multiblock.src.utils import dict_fault_ratio_fault
# from Article_journal_multiblock.src.utils import dict_fault_ratio_false_faults_found

def load_TEP_data():
    nb_simu_validation=50


    Testing_df=pd.DataFrame()

    for i in range(10):
        Testing_df=pd.concat([Testing_df,pd.read_parquet("../Data/TEP_Faulty_Testing_parquet_"+str(i))])

    Training_df=pd.read_parquet("../Data/TEP_FaultFree_Training_parquet")


    cols_feature=[x for x in Training_df.columns.tolist() if 'x' in x]
    Validation_df=Training_df.loc[Training_df.simulationRun<=nb_simu_validation]

    Training_df=Training_df.loc[Training_df.simulationRun>nb_simu_validation]

    
    return(Training_df,Testing_df,Validation_df,cols_feature)



def load_TEP_data_test():
    nb_simu_validation=1


    Testing_df=pd.DataFrame()

    for i in range(10):
        Testing_df=pd.concat([Testing_df,pd.read_parquet("../Data/TEP_Faulty_Testing_parquet_"+str(i))])
    Testing_df=Testing_df.loc[Testing_df.simulationRun<=1]

    Training_df=pd.read_parquet("../Data/TEP_FaultFree_Training_parquet")


    cols_feature=[x for x in Training_df.columns.tolist() if 'x' in x]
    Validation_df=Training_df.loc[Training_df.simulationRun<=nb_simu_validation]

    Training_df=Training_df.loc[Training_df.simulationRun>nb_simu_validation]
    


    Training_df=Training_df.loc[Training_df.simulationRun<=2]
    
    return(Training_df,Testing_df,Validation_df,cols_feature)
