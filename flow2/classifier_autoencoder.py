# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 01:48:41 2019

@author: Pronaya
"""
import pandas as pd
import numpy as np
from scipy import stats
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve, accuracy_score
from sklearn.metrics import recall_score, classification_report, auc, roc_curve
from sklearn.metrics import precision_recall_fscore_support, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pylab import rcParams
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
from scipy.signal import savgol_filter
from scipy.ndimage.filters import gaussian_filter
import category_encoders as ce

#path = "K:/data science course data"
path = "/media/pronaya/My Passport1/data science course data/test_data/flow2"
sample_size = 0.0001
t_size = 0.2

df = pd.read_csv(path+"/Test_day_9_10000N200AN.txt_reduced", header=None)
df.columns = ['time_sec', 'src_dom', 'des_dom', 'src', 'des', 'auth', 'class']
df.drop(["class"], axis = 1, inplace = True) 
df.head(n=5) #just to check you imported the dataset properly

max_sec = max(df['time_sec'])
min_sec = min(df['time_sec'])

df_redteam = pd.read_table(path+"/Test_day_9_10000N200AN.txt_compromised", sep=",", header=None)
df_redteam = df_redteam[[0, 2, 3, 4]]
df_redteam.columns = ['time_sec', 'des_dom', 'src', 'des' ]
df_redteam['class'] = 1
df_redteam_selected = df_redteam[(df_redteam['time_sec']>=min_sec) & (df_redteam['time_sec']<=max_sec)]

########### rectifying redteam data ####################################
df_anomaly_labeled = pd.merge(df, df_redteam_selected,  how='left', 
                          left_on=['time_sec', 'des_dom', 'src', 'des'], 
                          right_on = ['time_sec', 'des_dom', 'src', 'des'])

df_tmp2 = df_anomaly_labeled[(df_anomaly_labeled['class']==1)]
#df_redteam_selected['class']=0
df_tmp3 = pd.merge(df_redteam_selected, df_tmp2,  how='left', left_on=['time_sec', 'des_dom', 'src', 'des'], right_on = ['time_sec', 'des_dom', 'src', 'des'])
df_redteam_not_available = df_tmp3[df_tmp3['auth'].isnull()].drop(['src_dom', 'class_x', 'auth', 'class_y'], axis=1) 
df_redteam_available = df_tmp3[df_tmp3['auth']==1].drop(['src_dom', 'class_x', 'auth', 'class_y'], axis=1) 
df_redteam_available['class'] = 1
########################################################################
############# partitioning normal and anomaly data #####################
df_anomaly_labeled = pd.merge(df, df_redteam_available,  how='left', 
                          left_on=['time_sec', 'des_dom', 'src', 'des'], 
                          right_on = ['time_sec', 'des_dom', 'src', 'des'])
df_anomaly = df_anomaly_labeled[df_anomaly_labeled['class']==1].reset_index().drop(['index'], axis=1)
df_normal = df_anomaly_labeled[df_anomaly_labeled['class'].isnull()].reset_index().drop(['index'], axis=1)
df_normal['class'] = -1
########################################################################
############## Create test dataset #####################################
train, test_normal = train_test_split(df_normal, test_size=t_size)
test_mix = pd.concat([test_normal, df_anomaly]).reset_index().drop(['index'], axis=1)

########################################################################
df_normal_sample = train.sample(frac=sample_size)
df_normal_sample.head(n=5)

Y_train = df_normal_sample['class']
X_train = df_normal_sample.drop('class', axis = 1)

test_mix_copy = test_mix.copy()
#test_mix_copy['class'] = -1
Y_test = test_mix_copy['class']
X_test = test_mix.drop('class', axis = 1)


ce_mestimator = ce.MEstimateEncoder()
X_train_encode = ce_mestimator.fit_transform(X_train, Y_train)
X_test_encode = ce_mestimator.fit_transform(X_test, Y_test)

#X_train[:, 1] = LabelEncoder.fit_transform(X_train[:, 1])
#X_test_encode = LabelEncoder.fit_transform(X_test)



train_x = StandardScaler().fit_transform(X_train_encode)
train_x.shape
test_x = StandardScaler().fit_transform(X_test_encode)
test_x.shape


#Autoencoder Layer Structure and Parameters
nb_epoch = 16
batch_size = 100
input_dim = train_x.shape[1] #num of columns, 6
encoding_dim = 30
hidden_dim = int(encoding_dim / 3) #i.e. 5
learning_rate = 1e-7

input_layer = Input(shape=(input_dim, ))
encoder = Dense(encoding_dim, activation="tanh", activity_regularizer=regularizers.l1(learning_rate))(input_layer)
encoder = Dense(hidden_dim, activation="relu")(encoder)
decoder = Dense(hidden_dim, activation='tanh')(encoder)
decoder = Dense(input_dim, activation='relu')(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)

#Model Training and Logging

autoencoder.compile(metrics=['accuracy'],
                    loss='mean_squared_error',
                    optimizer='adam')

cp = ModelCheckpoint(filepath=path+"/autoencoder_anomaly2.h5",
                               save_best_only=True,
                               verbose=0)

tb = TensorBoard(log_dir=path+'/logs',
                histogram_freq=0,
                write_graph=True,
                write_images=True)

history = autoencoder.fit(train_x, train_x,
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    shuffle=True,
                    #validation_split = 0.1,
                    validation_data=(test_x, test_x),
                    verbose=1,
                    callbacks=[cp, tb]).history

autoencoder = load_model(path+'/autoencoder_anomaly2.h5')
train_x_predictions = autoencoder.predict(train_x)
mse = np.mean(np.power(train_x - train_x_predictions, 2), axis=1)
error_train_df = pd.DataFrame({'reconstruction_error': mse})
#error_train_df['reconstruction_error'][0]=0
description = error_train_df.describe()
threshold = description.iloc[1, 0]


test_x_predictions = autoencoder.predict(test_x)
mse = np.mean(np.power(test_x - test_x_predictions, 2), axis=1)
error_test_df = pd.DataFrame({'reconstruction_error': mse})
#error_test_df['reconstruction_error'][0]=0
error_test_df.describe()

error_test_df[error_test_df['reconstruction_error']>threshold] = 1
error_test_df[error_test_df['reconstruction_error']<=threshold] = 0
plt.plot(error_test_df['reconstruction_error'])
plt.show()

test_mix['predicted'] = error_test_df['reconstruction_error']
test_mix.loc[test_mix['class']==-1, 'class'] = 0


def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1
    return(TP, FP, TN, FN)


acc = accuracy_score(test_mix['class'], test_mix['predicted'])
print(acc)
[TP, FP, TN, FN] = perf_measure(test_mix['class'], test_mix['predicted'])
print([TP, FP, TN, FN])
print((FP/len(test_mix[test_mix['class']==0]))*100)
print((FN/len(test_mix[test_mix['class']==1]))*100)

print((TP+TN)/(len(test_mix)))

print("true positive rate: "+ str((TP/(TP+FN))*100))
print("true negative rate: "+ str((TN/(TN+FP))*100))
print("false positive rate: "+ str((FP/(FP+TN))*100))
print("false negative rate: "+ str((FN/(FN+TP))*100))


