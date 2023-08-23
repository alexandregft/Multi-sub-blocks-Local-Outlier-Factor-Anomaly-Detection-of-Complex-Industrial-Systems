from sklearn.preprocessing import StandardScaler
import numpy as np

def compute_t2_hotelling(X_train, X_test):
    # Compute the mean vector and covariance matrix from X_train
    
    
    scaler=StandardScaler()
    
    return t2_hotelling_scores





# Define the function 'train_hotelling_model'
def train_hotelling_model(Training_df,cols_feature,list_group_features):
    
    # Create empty lists to store the scalers, mean matrices, and inverse covariance matrices
    scaler_list=[]
    mean_matrix_list=[]
    inv_covariance_matrix_list=[]
    
    # Convert the training data into a numpy array
    X_train=Training_df[cols_feature].to_numpy()
    
    # Loop through each group of features
    for features_item in range(len(list_group_features)):
        
        # Create a StandardScaler object
        scaler=StandardScaler()

        # Select a subset of the training data and apply the scaler
        x_train_transform=X_train[:int(len(X_train)*0.1),list_group_features[features_item]]
        x_train_transform=scaler.fit_transform(x_train_transform)
        
        # Append the scaler object to the scaler list
        scaler_list.append(scaler)
        
        # Compute the mean vector and covariance matrix for the transformed data
        mean_vector = np.mean(x_train_transform, axis=0)
        covariance_matrix = np.cov(x_train_transform.T)

        # Compute the inverse covariance matrix and append it to the list
        inv_covariance_matrix = np.linalg.inv(covariance_matrix+1e-12)
        inv_covariance_matrix_list.append(inv_covariance_matrix)
            
        # Append the mean vector to the mean matrix list
        mean_matrix_list.append(mean_vector)
    
    # Return the lists of scalers, mean matrices, and inverse covariance matrices
    return(scaler_list,mean_matrix_list,inv_covariance_matrix_list)

def train_hotelling_model_real_data(X_train,list_group_features):
    
    # Create empty lists to store the scalers, mean matrices, and inverse covariance matrices
    scaler_list=[]
    mean_matrix_list=[]
    inv_covariance_matrix_list=[]
    
  
    # Loop through each group of features
    for features_item in range(len(list_group_features)):
        
        # Create a StandardScaler object

        # Select a subset of the training data and apply the scaler
        x_train_transform=X_train[:,list_group_features[features_item]]
        # Append the scaler object to the scaler list
        
        # Compute the mean vector and covariance matrix for the transformed data
        mean_vector = np.mean(x_train_transform, axis=0)
        covariance_matrix = np.cov(x_train_transform.T)
        # Compute the inverse covariance matrix and append it to the list
        if len(list_group_features[features_item])==1:
            inv_covariance_matrix=1/(covariance_matrix+1e-12)
        else:
            inv_covariance_matrix = np.linalg.inv(covariance_matrix+1e-12)
        inv_covariance_matrix_list.append(inv_covariance_matrix)
            
        # Append the mean vector to the mean matrix list
        mean_matrix_list.append(mean_vector)
    
    # Return the lists of scalers, mean matrices, and inverse covariance matrices
    return(mean_matrix_list,inv_covariance_matrix_list)


def compute_baysesian_inference_strategy(treshold,T2,alpha_item):
    
    
    p_xb_F=np.exp(-treshold/T2)
    p_xb_N=np.exp(-T2/treshold)
    p_xb=p_xb_N*alpha_item+(1-alpha_item)*p_xb_F
    P_F_xb=p_xb_F*(1-alpha_item)/(p_xb)
    
    
    return(p_xb_F,p_xb_N,p_xb,P_F_xb)



def BIC(list_group_features,T2_tab,threshold_list,alpha_item):
    """ Compute the Bayesian Inference Combination provided in [1]
    
    
    Parameters:
    
    -----------------------------
    
    list_group_features: List of the different group of features.
    Features are denoted using their index position in the cols_feature list.
    
    T2_tab:
    
    threshold_list:
    
    alpha_item:
    
    
    """
    BIC=0
    BIC_dnom=0
    for i in range(len(list_group_features)):
        p_xb_F,p_xb_N,p_xb,P_F_xb=compute_baysesian_inference_strategy(threshold_list[i],T2_tab[i,:],alpha_item)

        BIC=BIC+p_xb_F*P_F_xb
        BIC_dnom=BIC_dnom+p_xb_F
    return(BIC/BIC_dnom)



def generate_validation_results(parameters_trained_hotelling, Training_df, Validation_df, cols_feature, list_group_features, q):
    # Unpack parameters
    scaler_list, mean_matrix_list, inv_covariance_matrix_list = parameters_trained_hotelling
    
    # Convert validation and training dataframes to numpy arrays
    X_val = Validation_df[cols_feature].to_numpy()
    X_train = Training_df[cols_feature].to_numpy()
    
    # Initialize matrices for T2 Hotelling scores
    t2_hotelling_scores_val = np.zeros((len(list_group_features), len(X_val)))
    t2_hotelling_scores_train = np.zeros((len(list_group_features), len(X_train)))

    for features_item in range(len(list_group_features)):
        # Retrieve scaler, mean vector and inverse covariance matrix for current group of features
        scaler = scaler_list[features_item]
        X_val_item = X_val[:, list_group_features[features_item]]
        
        X_val_item=scaler.transform(X_val_item)

        mean_vector = mean_matrix_list[features_item]
        inv_covariance_matrix = inv_covariance_matrix_list[features_item]

        # Compute the T2 Hotelling score for each observation in X_test and X_train
        t2_hotelling_scores_item_val = np.dot((X_val_item - mean_vector), inv_covariance_matrix) * (X_val_item - mean_vector)
        t2_hotelling_scores_val[features_item, :] = np.sum(t2_hotelling_scores_item_val, axis=1)

        X_train_item = X_train[:, list_group_features[features_item]]
        
        
        X_train_item=scaler.transform(X_train_item)

        t2_hotelling_scores_item_train = np.dot((X_train_item - mean_vector), inv_covariance_matrix) * (X_train_item - mean_vector)
        t2_hotelling_scores_train[features_item, :] = np.sum(t2_hotelling_scores_item_train, axis=1)

    # Calculate T2 Hotelling score threshold for training data and use it to calculate BIC scores
    Treshold_T2_list = np.percentile(t2_hotelling_scores_train, (1-q)*100, axis=1)
    treshold_list_bic = np.percentile(BIC(list_group_features, t2_hotelling_scores_train, Treshold_T2_list, alpha_item=q), (1-q)*100)

    # Calculate BIC scores for validation data and use the threshold to make predictions
    results = BIC(list_group_features, t2_hotelling_scores_val, Treshold_T2_list, alpha_item=q)
    y_pred = results > treshold_list_bic

    # Return the proportion of positive predictions
    return np.sum(y_pred)/y_pred.size






def generate_validation_results_real_data(parameters_trained_hotelling, X_train, X_val, list_group_features, q):
    # Unpack parameters
    mean_matrix_list, inv_covariance_matrix_list = parameters_trained_hotelling
    
  
    # Initialize matrices for T2 Hotelling scores
    t2_hotelling_scores_val = np.zeros((len(list_group_features), len(X_val)))
    t2_hotelling_scores_train = np.zeros((len(list_group_features), len(X_train)))

    for features_item in range(len(list_group_features)):
        # Retrieve scaler, mean vector and inverse covariance matrix for current group of features
        X_val_item = X_val[:, list_group_features[features_item]]
        mean_vector = mean_matrix_list[features_item]
        inv_covariance_matrix = inv_covariance_matrix_list[features_item]
        # Compute the T2 Hotelling score for each observation in X_test and X_train
        t2_hotelling_scores_item_val = np.dot((X_val_item - mean_vector), inv_covariance_matrix) * (X_val_item - mean_vector)
        t2_hotelling_scores_val[features_item, :] = np.sum(t2_hotelling_scores_item_val, axis=1)

        X_train_item = X_train[:, list_group_features[features_item]]
        t2_hotelling_scores_item_train = np.dot((X_train_item - mean_vector), inv_covariance_matrix) * (X_train_item - mean_vector)
        t2_hotelling_scores_train[features_item, :] = np.sum(t2_hotelling_scores_item_train, axis=1)

    # Calculate T2 Hotelling score threshold for training data and use it to calculate BIC scores
    Treshold_T2_list = np.percentile(t2_hotelling_scores_train, (1-q)*100, axis=1)
    treshold_list_bic = np.percentile(BIC(list_group_features, t2_hotelling_scores_train, Treshold_T2_list, alpha_item=q), (1-q)*100)

    # Calculate BIC scores for validation data and use the threshold to make predictions
    results = BIC(list_group_features, t2_hotelling_scores_val, Treshold_T2_list, alpha_item=q)
    y_pred = results > treshold_list_bic

    # Return the proportion of positive predictions
    return y_pred






def generate_test_results_real_data(parameters_trained_hotelling, X_train,X_val,X_test, list_group_features, q):
    # Unpack parameters
    mean_matrix_list, inv_covariance_matrix_list = parameters_trained_hotelling
    
    # Initialize matrices for T2 Hotelling scores
    t2_hotelling_scores_train = np.zeros((len(list_group_features), len(X_train)))

    for features_item in range(len(list_group_features)):
        # mean vector and inverse covariance matrix for current group of features       
        mean_vector = mean_matrix_list[features_item]
        inv_covariance_matrix = inv_covariance_matrix_list[features_item]
        X_train_item = X_train[:, list_group_features[features_item]]
        t2_hotelling_scores_item_train = np.dot((X_train_item - mean_vector), inv_covariance_matrix) * (X_train_item - mean_vector)
        t2_hotelling_scores_train[features_item, :] = np.sum(t2_hotelling_scores_item_train, axis=1)

    # Calculate T2 Hotelling score threshold for training data and use it to calculate BIC scores
    Treshold_T2_list = np.percentile(t2_hotelling_scores_train, (1-q)*100, axis=1)
    treshold_list_bic = np.percentile(BIC(list_group_features, t2_hotelling_scores_train, Treshold_T2_list, alpha_item=q), (1-q)*100)
    detected_issues=[]
    t2_hotelling_scores_test = np.zeros((len(list_group_features), len(X_test)))

    for features_item in range(len(list_group_features)):
        mean_vector = mean_matrix_list[features_item]
        inv_covariance_matrix = inv_covariance_matrix_list[features_item]
        x_test_item = X_test[:, list_group_features[features_item]]
        t2_hotelling_scores_item_test = np.dot((x_test_item - mean_vector), inv_covariance_matrix) * (x_test_item - mean_vector)
        t2_hotelling_scores_test[features_item, :] = np.sum(t2_hotelling_scores_item_test, axis=1)

    results = BIC(list_group_features, t2_hotelling_scores_test, Treshold_T2_list, alpha_item=q)
    

    y_pred = results > treshold_list_bic

    
    return(y_pred,y_pred)




def generate_test_results(parameters_trained_hotelling, Training_df, Testing_df, cols_feature, list_group_features, q):
    # Unpack parameters
    scaler_list, mean_matrix_list, inv_covariance_matrix_list = parameters_trained_hotelling
    
    # Convert test and training dataframes to numpy arrays
    X_train = Training_df[cols_feature].to_numpy()
    

    
    # Initialize matrices for T2 Hotelling scores
    t2_hotelling_scores_train = np.zeros((len(list_group_features), len(X_train)))

    for features_item in range(len(list_group_features)):
        # Retrieve scaler, mean vector and inverse covariance matrix for current group of features
        scaler = scaler_list[features_item]
       

        
        mean_vector = mean_matrix_list[features_item]
        inv_covariance_matrix = inv_covariance_matrix_list[features_item]
        X_train_item = X_train[:, list_group_features[features_item]]
        X_train_item=scaler.transform(X_train_item)

        
        t2_hotelling_scores_item_train = np.dot((X_train_item - mean_vector), inv_covariance_matrix) * (X_train_item - mean_vector)
        t2_hotelling_scores_train[features_item, :] = np.sum(t2_hotelling_scores_item_train, axis=1)

    # Calculate T2 Hotelling score threshold for training data and use it to calculate BIC scores
    Treshold_T2_list = np.percentile(t2_hotelling_scores_train, (1-q)*100, axis=1)
    treshold_list_bic = np.percentile(BIC(list_group_features, t2_hotelling_scores_train, Treshold_T2_list, alpha_item=q), (1-q)*100)
    detected_issues=[]
    detected_list_glob=[]
    list_tab_res_fault=[]
    n_simu=int(max(Testing_df['simulationRun']))
    len_sample=int(max(Testing_df['sample']))
    Tab_all_results=np.zeros((max(Testing_df['faultNumber'].unique()),n_simu,len_sample))
    
    
    n_fault=max(Testing_df['faultNumber'].unique())
    
    for fault_num in range(1,n_fault+1):

    
        list_tab_res_i=[]
        x_test=Testing_df
        x_test=x_test.loc[x_test['faultNumber']==fault_num]
        issues_tab=np.zeros((len(x_test)//len_sample,len_sample))
        x_test=x_test[cols_feature].to_numpy()
        t2_hotelling_scores_test = np.zeros((len(list_group_features), len(x_test)))
        list_tab_res_i=[]

        for features_item in range(len(list_group_features)):
            # Retrieve scaler, mean vector and inverse covariance matrix for current group of features
            scaler = scaler_list[features_item]
            



            mean_vector = mean_matrix_list[features_item]
            inv_covariance_matrix = inv_covariance_matrix_list[features_item]

        
        
            x_test_item = x_test[:, list_group_features[features_item]]

            x_test_item=scaler.transform(x_test_item)
            t2_hotelling_scores_item_test = np.dot((x_test_item - mean_vector), inv_covariance_matrix) * (x_test_item - mean_vector)
            t2_hotelling_scores_test[features_item, :] = np.sum(t2_hotelling_scores_item_test, axis=1)

        results = BIC(list_group_features, t2_hotelling_scores_test, Treshold_T2_list, alpha_item=q)
        
        
        y_pred = results > treshold_list_bic
        tab_res_i=np.zeros(len_sample)




        for element in range(len(y_pred)//len_sample):
            tab_res_i=tab_res_i+y_pred[element*len_sample:(element+1)*len_sample]
            issues_tab[element,:]=issues_tab[element,:]+y_pred[element*len_sample:(element+1)*len_sample]
            Tab_all_results[fault_num-1,element,:]=y_pred[element*len_sample:(element+1)*len_sample]
        list_tab_res_fault.append(tab_res_i)
        detected_issues.append(issues_tab)
        
        
    
    return(Tab_all_results,detected_issues)
