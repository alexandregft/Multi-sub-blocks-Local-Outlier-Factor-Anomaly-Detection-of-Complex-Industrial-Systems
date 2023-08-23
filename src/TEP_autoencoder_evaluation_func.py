from keras import backend as K
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import f1_score

from sklearn.preprocessing import MinMaxScaler

import numpy as np


def encoder_simple(x_input,list_dimension_size):
    input_img = keras.Input(shape=(x_input.shape[1],))

    # This is the size of our encoded representations
    
    encoder_list=[]
    count=0
    for element in list_dimension_size:
        if count==0:
            encoder_list.append( layers.Dense(element, activation='relu')(input_img))
        else:
            encoder_list.append( layers.Dense(element, activation='relu')(encoder_list[-1]))

        count=count+1
    # "decoded" is the lossy reconstruction of the input
    print(count)
    
    count=count-2
    for element in list_dimension_size[:-1]:
        if count==-1:
            encoder_list.append(layers.Dense(x_input.shape[1], activation='relu')(encoder_list[-1]))
        else:
            encoder_list.append(layers.Dense(list_dimension_size[count],activation='relu')(encoder_list[-1]))
        count=count-1

    # This model maps an input to its reconstruction
    encoder_list.append(layers.Dense(x_input.shape[1], activation='sigmoid')(encoder_list[-1]))

    autoencoder = keras.Model(input_img, encoder_list[-1])
    
    return(autoencoder)


def vae(x_input,list_dim):

    latent_dim=list_dim[-1]
    
    original_dim=x_input.shape[1]
    encoder_inputs = keras.Input(shape=(x_input.shape[1],))

    # This is the size of our encoded representations
    
    count=0
    for element in list_dim[:-1]:
        if count==0:
            x= layers.Dense(element, activation='relu')(encoder_inputs)
        else:
            x= layers.Dense(element, activation='relu')(x)
        count=+1
    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                                  mean=0., stddev=0.1)
        return z_mean + K.exp(z_log_sigma) * epsilon
    
    z_mean = layers.Dense(latent_dim)(x)
    z_log_sigma = layers.Dense(latent_dim)(x)
    z = layers.Lambda(sampling)([z_mean, z_log_sigma])
        # Create encoder
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_sigma, z], name='encoder')

    latent_inputs = keras.Input(shape=(latent_dim,), name='z_sampling')
    count=len(list_dim)-1
    
    for element in list_dim[:-1]:
        print(count)
        if count==len(list_dim)-1:
            h= layers.Dense(list_dim[count], activation='relu')(latent_inputs)
        else:
            h= layers.Dense(list_dim[count], activation='relu')(h)
        count=count-1
    
    outputs = layers.Dense(x_input.shape[1], activation='sigmoid')(h)
    decoder = keras.Model(latent_inputs, outputs, name='decoder')

    # instantiate VAE model
    outputs = decoder(encoder(encoder_inputs)[2])
    vae = keras.Model(encoder_inputs, outputs, name='vae_mlp')

    reconstruction_loss = keras.losses.binary_crossentropy(encoder_inputs, outputs)
    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')    
    return(vae)

def generate_ae_res_val(Training_df, Validation_df,cols_feature,model, latent_dims, n_epochs, batch_size, q):
    
    
    if model=="VAE":
        
        # create VAE model
        model = vae(Training_df[cols_feature].to_numpy(), latent_dims)
    elif model =="AE":
        model=encoder_simple(Training_df[cols_feature].to_numpy(), latent_dims)

    # create scaler and fit to a subset of training data
    scaler = MinMaxScaler()
    scaler.fit(Training_df[cols_feature].to_numpy()[0:int(len(Training_df)*0.1)])

    # prepare training data and compile model
    x_train = Training_df[cols_feature].to_numpy()[int(len(Training_df)*0.1):]
    x_train = scaler.transform(x_train)
    model.compile(optimizer='adam', loss='mean_squared_error')

    # train VAE model
    model.fit(x_train, x_train,
                  epochs=n_epochs,
                  batch_size=batch_size,
                  shuffle=True,
                  verbose=False)

    # prepare validation data and make predictions
    x_val = Validation_df[cols_feature].to_numpy()
    x_val = scaler.transform(x_val)

    x_val_pred = model.predict(x_val, verbose=False)
    

    rmse_tot=((x_val_pred-x_val)**2).sum(axis=1)

    init_threshold={}
    init_threshold['down']=np.percentile(rmse_tot,(q)*100)
    init_threshold['up']=np.percentile(rmse_tot,(1-q)*100)
    y_pred=rmse_tot>init_threshold['up']
    return(np.sum(y_pred)/len(y_pred))




def generate_ae_res_val_real_data(X_train,X_val,X_test,model, latent_dims, n_epochs, batch_size, q):
    
    
    if model=="VAE":
        
        # create VAE model
        model = vae(X_train, latent_dims)
    elif model =="AE":
        model=encoder_simple(X_train, latent_dims)

    # create scaler and fit to a subset of training data
   

    # prepare training data and compile model
    x_train = X_train
    model.compile(optimizer='adam', loss='mean_squared_error')

    # train VAE model
    model.fit(x_train, x_train,
                  epochs=n_epochs,
                  batch_size=batch_size,
                  shuffle=True,
                  verbose=False)
    
    x_val = X_val

    x_val_pred = model.predict(x_val, verbose=False)
    

    rmse_tot=((x_val_pred-x_val)**2).sum(axis=1)

    init_threshold={}
    init_threshold['down']=np.percentile(rmse_tot,(q)*100)
    init_threshold['up']=np.percentile(rmse_tot,(1-q)*100)
    y_pred=rmse_tot>init_threshold['up']
    
    
#     y_true=Y_test
    
#     y_pred_tot=y_pred

#     f1 = f1_score(y_true, y_pred_tot, average='macro')
    
    
#     n_class_1 = np.sum(Y_test == 1)

#     # Count the number of instances of class 1 that were correctly predicted
#     n_class_1_retrieved = np.sum((Y_test == 1) & (y_pred_tot == 1))

#     # Compute the ratio of class 1 retrieved
#     ratio_class_1_retrieved = n_class_1_retrieved / n_class_1

    return(y_pred)


def generate_res_ae_real_data(X_train,X_val,X_test,model, latent_dims, n_epochs, batch_size, q):


    if model=="VAE":
        
        # create VAE model
        model = vae(X_train, latent_dims)
    elif model =="AE":
        model=encoder_simple(X_train, latent_dims)

    # create scaler and fit to a subset of training data
    
    # prepare training data and compile model
    x_train = X_train
    model.compile(optimizer='adam', loss='mean_squared_error')

    # train VAE model
    model.fit(x_train, x_train,
                  epochs=n_epochs,
                  batch_size=batch_size,
                  shuffle=True,
                  verbose=False)
    x_val = X_val

    x_val_pred = model.predict(x_val, verbose=False)
    

    rmse_tot=((x_val_pred-x_val)**2).sum(axis=1)

    init_threshold={}
    init_threshold['down']=np.percentile(rmse_tot,(q)*100)
    init_threshold['up']=np.percentile(rmse_tot,(1-q)*100)
    
    x_test=X_test
    x_test_pred=model.predict(X_test, verbose=False)
    rmse_tot=((x_test_pred-x_test)**2).sum(axis=1)

    
    y_pred=rmse_tot>init_threshold['up']
    
    
    
    y_pred_tot=y_pred
    return(y_pred_tot,y_pred_tot)



def generate_res_ae(Training_df,Testing_df, Validation_df,cols_feature,model, latent_dims, n_epochs, batch_size, q):
    
    if model=="VAE":
        # create VAE model
        model = vae(Training_df[cols_feature].to_numpy(), latent_dims)
    elif model=="AE":
        model=encoder_simple(Training_df[cols_feature].to_numpy(), latent_dims)
    

    # create scaler and fit to a subset of training data
    scaler = MinMaxScaler()
    scaler.fit(Training_df[cols_feature].to_numpy()[0:int(len(Training_df)*0.1)])

    # prepare training data and compile model
    x_train = Training_df[cols_feature].to_numpy()[int(len(Training_df)*0.1):]
    x_train = scaler.transform(x_train)
    model.compile(optimizer='adam', loss='mean_squared_error')

    # train VAE model
    model.fit(x_train, x_train,
                  epochs=n_epochs,
                  batch_size=batch_size,
                  shuffle=True,
                  verbose=False)

    # prepare validation data and make predictions
    x_val = Validation_df[cols_feature].to_numpy()
    x_val = scaler.transform(x_val)

    x_val_pred = model.predict(x_val, verbose=False)
    

    rmse_tot=((x_val_pred-x_val)**2).sum(axis=1)

    init_threshold={}
    init_threshold['down']=np.percentile(rmse_tot,(q)*100)
    init_threshold['up']=np.percentile(rmse_tot,(1-q)*100)
    
    
    
    
    
    detected_issues=[]
    detected_list_glob=[]
    list_tab_res_fault=[]
    n_simu=int(max(Testing_df['simulationRun']))
    len_sample=int(max(Testing_df['sample']))

    Tab_all_results=np.zeros((max(Testing_df['faultNumber'].unique()),n_simu,len_sample))

    for fault_num in range(1,max(Testing_df['faultNumber'].unique())+1):

        list_tab_res_i=[]
        x_test=Testing_df
        x_test=x_test.loc[x_test['faultNumber']==fault_num]
        issues_tab=np.zeros((len(x_test)//len_sample,len_sample))

        x_test=x_test[cols_feature].to_numpy()
        x_test=scaler.transform(x_test)
        x_test_pred=model.predict(x_test,verbose=False)
        rmse_test=((x_test-x_test_pred)**2).sum(axis=1)
        index_1=(rmse_test>init_threshold['up'])
        index_zero=(rmse_test>init_threshold['up'])==0


        y_pred=rmse_test
        y_pred[index_zero]=0
        y_pred[index_1]=1

        tab_res_i=np.zeros(len_sample)

        for element in range(len(y_pred)//len_sample):
            tab_res_i=tab_res_i+y_pred[element*len_sample:(element+1)*len_sample]
            issues_tab[element,:]=issues_tab[element,:]+y_pred[element*len_sample:(element+1)*len_sample]
            Tab_all_results[fault_num-1,element,:]=y_pred[element*len_sample:(element+1)*len_sample]
        list_tab_res_fault.append(tab_res_i)
        detected_issues.append(issues_tab)

    return(Tab_all_results,detected_issues)