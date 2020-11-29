# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 02:19:52 2019

@author: Pronaya
"""
# import packages
# matplotlib inline
import pandas as pd
import numpy as np
from scipy import stats
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import recall_score, classification_report, auc, roc_curve
from sklearn.metrics import precision_recall_fscore_support, f1_score
from sklearn.preprocessing import StandardScaler
from pylab import rcParams
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
from scipy.signal import savgol_filter
from scipy.ndimage.filters import gaussian_filter

#set random seed and percentage of test data
RANDOM_SEED = 314 #used to help randomly select the data points
TEST_PCT = 0.2 # 20% of the data

#set up graphic style in this case I am using the color scheme from xkcd.com
rcParams['figure.figsize'] = 14, 8.7 # Golden Mean
LABELS = ["Normal","Fraud"]
col_list = ["cerulean","scarlet"]# https://xkcd.com/color/rgb/
sns.set(style='white', font_scale=1.75, palette=sns.xkcd_palette(col_list), color_codes=False)

path = "K:/data science course data"
df = pd.read_csv(path+"/service_per_min.csv")
df.head(n=5) #just to check you imported the dataset properly

df.isnull().values.any()

def day_to_min(day=3):
    start_min = day*24*60+1
    end_min = start_min + 24*60
    return (start_min, end_min)


df_norm = df[(df['time_min']>=day_to_min(7)[0]) & (df['time_min']<day_to_min(7)[1])]
df_norm['time_min_standard'] = df_norm['time_min']-min(df_norm['time_min'])+1
plt.plot(df_norm['time_min_standard'], df_norm['count'])
df_norm = df_norm.reset_index()
df_norm = df_norm.drop(['index', 'sec_from', 'sec_to', 'time_min'], axis=1) 


df_test = df[(df['time_min']>=day_to_min(49)[0]) & (df['time_min']<day_to_min(49)[1])]
df_test['time_min_standard'] = df_test['time_min']-min(df_test['time_min'])+1
plt.plot(df_test['time_min_standard'], df_test['count'])
df_test = df_test.reset_index()
df_test = df_test.drop(['index', 'sec_from', 'sec_to', 'time_min'], axis=1) 


train_x = StandardScaler().fit_transform(df_norm)
#train_x = train_x.values #transform to ndarray
train_x.shape

test_x = StandardScaler().fit_transform(df_test)
#train_x = train_x.values #transform to ndarray
test_x.shape


#Autoencoder Layer Structure and Parameters
nb_epoch = 500
batch_size = 100
input_dim = train_x.shape[1] #num of columns, 30
encoding_dim = 30
hidden_dim = int(encoding_dim / 6) #i.e. 7
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

cp = ModelCheckpoint(filepath=path+"/autoencoder_fraud.h5",
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
                    validation_split = 0.1,
                    #validation_data=(test_x, test_x),
                    verbose=1,
                    callbacks=[cp, tb]).history

autoencoder = load_model(path+'/autoencoder_fraud.h5')
train_x_predictions = autoencoder.predict(train_x)
mse = np.mean(np.power(train_x - train_x_predictions, 2), axis=1)
error_train_df = pd.DataFrame({'reconstruction_error': mse})
error_train_df['reconstruction_error'][0]=0
error_train_df.describe()
threshold = max(error_train_df['reconstruction_error'])/3


test_x_predictions = autoencoder.predict(test_x)
mse = np.mean(np.power(test_x - test_x_predictions, 2), axis=1)
error_test_df = pd.DataFrame({'reconstruction_error': mse})
error_test_df['reconstruction_error'][0]=0
error_test_df.describe()

def applyThreshold(x):
    if(x<threshold):
        return 0
    else:
        return 1
error_test_df['reconstruction_error_threshold'] = error_test_df['reconstruction_error'].apply(applyThreshold)


error_test_df['reconstruction_error_norm'] = savgol_filter(error_test_df['reconstruction_error'], 51, 3)
plt.plot(error_test_df['reconstruction_error'])
plt.show()
plt.plot(error_test_df['reconstruction_error_norm'])
plt.show()
plt.plot(error_test_df['reconstruction_error_threshold'])
plt.show()


df_test['count_anomaly'] = df_test['count']*error_test_df['reconstruction_error_threshold']
plt.plot(df_test['time_min_standard'], df_test['count'], '-b', label='Normal')
plt.plot(df_test['time_min_standard'], df_test['count_anomaly'], 'or', label='Anomaly')
plt.xlabel('Minutes')
plt.ylabel("Counts")
plt.legend()
plt.show()
plt.savefig(path+'/anomaly.png', format='png', dpi=600)




