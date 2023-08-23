from sklearn.model_selection import train_test_split

import os
import pandas as pd
import numpy as np
import pymit

def get_file_paths(folder_path):
    """
    Returns a list of all file paths in the given folder path.
    """
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths


def load_real_data(version_name,files_training,files_limits):
    Test_value_df=pd.read_csv([x for x in files_training if version_name in x][0])
    Limits_df=pd.read_csv([x for x in files_limits if version_name in x][0])
    cols_features=[x for x in Test_value_df.columns.tolist() if "mes" in x]
    Test_value_df=Test_value_df.dropna(subset=cols_features,how="all")
    # Calculate the percentage of NaN values in each column
    nan_percentage = Test_value_df.isna().sum() / len(Test_value_df)

    # Select columns with less than or equal to 10% NaN values
    selected_cols = Test_value_df.columns[nan_percentage <= 0.1]

    # Drop columns with more than 10% NaN values
    Test_value_df = Test_value_df[selected_cols]

    Test_value_df=Test_value_df.reset_index(drop=True)
    Test_value_df=Test_value_df.sort_values('Date')
    cols_features=[x for x in Test_value_df.columns.tolist() if "mes" in x]
    Test_value_df=Test_value_df.dropna(ignore_index=True)
    return(Test_value_df,Limits_df,cols_features)


def check_test_values(test_values, limits):
    # Create empty lists to hold the test names
    out_of_limit_tests = []
    valid_limit_tests = []

    # Iterate over the columns in the test_values dataframe
    for column in test_values.columns:
        try:
            # Extract the test name from the column name
            test = column[:-3]

            # Get the test value and limit for the current test
            test_value = test_values[column]
            test_limits = limits.filter(regex=f'{test}lim')

            # Check if any of the test values are out of limits
            out_of_limit_mask = np.sum(((test_value < test_limits[f'{test}lim_b'].iloc[0]) | 
                                (test_value > test_limits[f'{test}lim_h'].iloc[0])))
            
            out_of_limit_mask_1=(out_of_limit_mask/len(test_value))<0.02
            out_of_limit_mask_2=(out_of_limit_mask/len(test_value))>0
            # Check if the test limits are valid
            valid_limits_mask = ((test_limits[f'{test}lim_h'].iloc[0] > 0) & (test_limits[f'{test}lim_h'].iloc[0] < 1e8) & 
                                (test_limits[f'{test}lim_b'].iloc[0] > 0) & (test_limits[f'{test}lim_b'].iloc[0] < 1e8)).all()

            # Add the test to the appropriate list if it meets the criteria
            
           
            if out_of_limit_mask_1:
                if valid_limits_mask:
                    if out_of_limit_mask_2:
                        out_of_limit_tests.append(test)
                    else:
                        valid_limit_tests.append(test)
        except:
            0

    return out_of_limit_tests, valid_limit_tests


import random

def select_random_tests(test_values, limits, n):
    # Compute the out-of-limit and valid-limit test lists
    out_of_limit_tests, valid_limit_tests = check_test_values(test_values, limits)

    # Set a random seed for reproducibility
    random.seed(123)

    # Select n tests randomly from the valid-limit test list

    if n-len(out_of_limit_tests)>0:
        random_tests = random.sample([test for test in valid_limit_tests if test not in out_of_limit_tests], n-len(out_of_limit_tests))
        random_tests=out_of_limit_tests+random_tests
    else:
        random_tests=out_of_limit_tests
    # Get the subset of the test values dataframe corresponding to the random tests
    random_test_values = test_values.filter([f'{test}mes' for test in random_tests])
    
    return random_tests, random_test_values


def create_label(random_test_values,Limits_df):
    # Create example test values DataFrame and limits DataFrame

    # Create empty labels Series
    Y_label=np.zeros(len(random_test_values))
    # Loop over test columns and compare to limits
    for test_col in random_test_values.columns:
        test=test_col[:-3]
        test_values_col = random_test_values[test_col]
        limit_h_col = float(Limits_df[f'{test}lim_h'])
        limit_b_col = float(Limits_df[f'{test}lim_b'])
        mask = ((test_values_col > limit_h_col) | (test_values_col < limit_b_col))
        Y_label[mask] = 1

    # Add labels to test values DataFrame
    random_test_values['label'] = Y_label
    return(random_test_values)


def create_train_test(random_test_values):
        
    # Split into train, validation, and test sets
    X_train = random_test_values.iloc[:int(len(random_test_values)*0.7), :]
    X_test = random_test_values.iloc[int(len(random_test_values)*0.7):, :]

    # Remove all observations with Y label == 1 from X_train and X_val, and add them to X_test
    X_test = pd.concat([X_test, X_train[X_train['label'] == 1]])
    
    X_train = X_train[X_train['label'] != 1]
    
    index_y_train=X_train.index.tolist()
    index_y_test=X_test.index.tolist()

    X_train=X_train.reset_index()
    X_test=X_test.reset_index()
    Y_test = X_test['label']
    Y_train = X_train['label']



    # Extract features and convert to numpy arrays
    features = [x for x in X_train.columns if "mes" in x]
    X_train = X_train[features].to_numpy()
    X_test = X_test[features].to_numpy()


    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    return(X_train,Y_train,X_test,Y_test,index_y_test)





def create_train_test_val(random_test_values):
        
    # Split into train, validation, and test sets
    size_split=0.8
    # X_train = random_test_values.iloc[:int(len(random_test_values)*size_split), :]
    # X_test = random_test_values.iloc[int(len(random_test_values)*size_split):, :]
    random_state=42
    X_train,X_test=train_test_split(random_test_values,  train_size=size_split, shuffle=True,random_state=random_state)


    # Remove all observations with Y label == 1 from X_train and X_val, and add them to X_test
    X_test = pd.concat([X_test, X_train[X_train['label'] == 1]])
    
    X_train = X_train[X_train['label'] != 1]
    
    index_y_train=X_train.index.tolist()
    index_y_test=X_test.index.tolist()

    X_train=X_train.reset_index()
    X_test=X_test.reset_index()
    Y_test = X_test['label']
    Y_train = X_train['label']



    # Extract features and convert to numpy arrays
    features = [x for x in X_train.columns if "mes" in x]
    X_train = X_train[features].to_numpy()
    X_test = X_test[features].to_numpy()
    
    
    X_train,X_val=train_test_split(X_train,  test_size=0.3, shuffle=True,random_state=random_state)

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    return(X_train,Y_train,X_test,Y_test,X_val,index_y_test)




def create_aja_matrix(X_train_array_normalised):
    
    Adja_maxtrix=np.zeros((X_train_array_normalised.shape[1],X_train_array_normalised.shape[1]))
    for feature_i in range(X_train_array_normalised.shape[1]):
        for feature_j in range(X_train_array_normalised.shape[1]):
            if feature_i<feature_j:
                Adja_maxtrix[feature_i,feature_j]=pymit.I(X_train_array_normalised[:, feature_i], X_train_array_normalised[:, feature_j], bins=[100, 100])
                Adja_maxtrix[feature_j,feature_i]=pymit.I(X_train_array_normalised[:, feature_i], X_train_array_normalised[:, feature_j], bins=[100, 100])
    return(Adja_maxtrix)

def create_normalized_adja_matrix(Adja_maxtrix):
    D_aja=np.zeros((len(Adja_maxtrix),len(Adja_maxtrix)))
    for i in range(len(Adja_maxtrix)):
        inter_calcul=np.sqrt(np.sum(Adja_maxtrix[i,:]))
        D_aja[i,i]=1/inter_calcul

    normalised_Adja_maxtrix=np.dot(Adja_maxtrix,np.dot(Adja_maxtrix,D_aja[i,i]))
    return(normalised_Adja_maxtrix)



def save_cluster_data(cluster_ids, costs, file_path):
    # create a dictionary from the given lists
    data = {"cluster_id": cluster_ids, "cost": costs}

    # create a pandas dataframe from the dictionary
    df = pd.DataFrame(data)

    # save the dataframe to the specified file path
    df.to_csv(file_path, index=False)