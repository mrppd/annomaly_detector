# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 04:43:43 2019

@author: Pronaya
"""
##https://stackoverflow.com/questions/29287224/pandas-read-in-table-without-headers


import pandas as pd
import numpy as np
import sys

import matplotlib
import seaborn
import matplotlib.dates as md
from matplotlib import pyplot as plt

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.covariance import EllipticEnvelope
#from pyemma import msm # not available on Kaggle Kernel
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from datetime import datetime, timedelta

contamination_rate = 0.003
#path = "K:/data science course data/test_data"
path = "/media/pronaya/My Passport1/data science course data/test_data/flow3"

"""
if(len(sys.argv)<1):
    exit(0)
day = int(sys.argv[1])
"""
day = 8
day=day+1

df = pd.read_csv(path+"/src_info_src_desTest_day_9_10000N200AN.txt_reduced.csv")

df_selected = df[["num_of_des", "total_failed", "con_per_des"]]

min_max_scaler = preprocessing.StandardScaler()
np_scaled = min_max_scaler.fit_transform(df_selected)
data = pd.DataFrame(np_scaled)
# train isolation forest 
model =  IsolationForest(contamination = contamination_rate, n_estimators=300, random_state=5, n_jobs=2, verbose=1)
model.fit(data)
# add the data to the main  
df['anomaly'] = pd.Series(model.predict(data))
df['anomaly'] = df['anomaly'].map( {1: 0, -1: 1} )
print(df['anomaly'].value_counts())


#df_sort = df.sort_values(['num_of_des', 'con_per_des'], ascending=[False, False])

#df_anomaly = df[df['anomaly']==1][['src', 'des_list', 'anomaly']]
df_anomaly = df[['src', 'des_list', 'anomaly']]

#spliting the des_list column
lst_col = 'des_list' 
x = df_anomaly.assign(**{lst_col:df_anomaly[lst_col].str.split(' ')})

df_anomaly = pd.DataFrame({col:np.repeat(x[col].values, x[lst_col].str.len())
                for col in x.columns.difference([lst_col])
            }).assign(**{lst_col:np.concatenate(x[lst_col].values)})[x.columns.tolist()]
df_anomaly.columns = ['src', 'des', 'anomaly']



#reading src_des for reference
df_src_des = pd.read_csv(path+"/src_desTest_day_9_10000N200AN.txt_reduced"+".csv")
df_src_des_s = df_src_des.groupby(['src','des'])['times', 'failed'].sum().reset_index()

#reading day_reduced data to get start and end time (in seconds)
df_day_reduced = pd.read_table(path+"/Test_day_9_10000N200AN.txt_reduced", sep=",", header=None)
max_sec = max(df_day_reduced[0])
min_sec = min(df_day_reduced[0])
print("From ", min_sec, "To: ", max_sec)


#adding times and failed src_des with processed src_info data
df_anomaly_new = pd.merge(df_anomaly, df_src_des_s,  how='left', left_on=['src','des'], right_on = ['src','des'])
df_anomaly_new['real_anomaly']=-1
df_anomaly_new['real_times']=0


#reading anomaly references for further analysis
df_redteam = pd.read_table(path+"/Test_day_9_10000N200AN.txt_compromised", sep=",", header=None)
df_redteam = df_redteam[[0, 2, 3, 4]]
df_redteam.columns = ['time_sec', 'des_domain', 'src', 'des' ]
df_redteam['time_day'] = df_redteam['time_sec']//(24*3600)
df_redteam['times']=1

df_redteam_sub = df_redteam.loc[(df_redteam['time_sec']>=min_sec) & (df_redteam['time_sec']<=max_sec)]
df_redteam_sub['real_anomaly_check']=0

df_redteam_sub = df_redteam_sub.groupby(['src','des'])['times', 'real_anomaly_check'].sum().reset_index()


#df_anomaly_new_with_redteam = pd.merge(df_anomaly_new, df_redteam_sub,  how='left', left_on=['src','des'], right_on = ['src','des'])

for j in range(len(df_redteam_sub)):
    df_anomaly_new.loc[(df_anomaly_new['src']==df_redteam_sub.iloc[j].src) &
                       (df_anomaly_new['des']==df_redteam_sub.iloc[j].des), 'real_anomaly']=1
    df_anomaly_new.loc[(df_anomaly_new['src']==df_redteam_sub.iloc[j].src) &
                       (df_anomaly_new['des']==df_redteam_sub.iloc[j].des), 'real_times']=df_redteam_sub.iloc[j, 2]
    detected = sum((df_anomaly_new['src']==df_redteam_sub.iloc[j].src)&(df_anomaly_new['des']==df_redteam_sub.iloc[j].des))
    df_redteam_sub.iloc[j, 3] = detected
    if(j%50==0):
        print(j)
df_anomaly_new.loc[(df_anomaly_new['real_anomaly']==-1), 'real_anomaly']=0


#calculating difference in connection count between day data and redteam data
df_anomaly_new['diff_times'] = df_anomaly_new['times'] - df_anomaly_new['real_times']

#separating anomaly only data 
df_anomaly_new_detected = df_anomaly_new[(df_anomaly_new['real_anomaly']==1)]
df_anomaly_new_detected = df_anomaly_new_detected[(df_anomaly_new_detected['anomaly']==1)]

df_anomaly_new_predicted = df_anomaly_new[(df_anomaly_new['anomaly']==1)]

#print("Day: "+str(day-1))
if(len(df_redteam_sub)>0):
    #acc = round((len(df_anomaly_new_detected)/len(df_redteam_sub))*100, 3)
    
    false_pos = sum(df_anomaly_new_predicted['diff_times'])
    print("Number of false positive: "+str(false_pos)+"/"+str(sum(df_anomaly_new['times'])))
    false_pos_rate = round(sum(df_anomaly_new_predicted['diff_times'])/sum(df_anomaly_new['times'])*100, 3)
    print("False positive rate: "+str(false_pos_rate))
    #false_neg = sum(df_redteam_sub[df_redteam_sub['real_anomaly_check']==0]['times'])
    #print("Number of false nagetive: "+ str(false_neg))
   
    true_pos_rate = round((len(df_anomaly_new_detected)/len(df_redteam_sub))*100, 3)
    print("True positive rate: "+str(true_pos_rate))
    
    false_neg_rate = 100-true_pos_rate #round(false_neg/sum(df_redteam_sub['times'])*100, 3)
    print("False negetive rate: "+str(false_neg_rate))
    true_neg_rate = 100 - false_pos_rate
    print("True negetive rate: "+str(true_neg_rate))
    acc = (true_pos_rate + true_neg_rate)/(true_pos_rate + true_neg_rate + false_pos_rate + false_neg_rate)
    print("Accuracy: "+str(round(acc*100, 3)))
    
    #print("##################Not Needed#########################")
    #false_pos_unique = len(df_anomaly_new_predicted)-len(df_anomaly_new_detected)
    #print("Number of false positive unique: "+str(false_pos_unique)+"/"+str(len(df_anomaly_new)))
    #false_pos_rate_unique = round((false_pos_unique/len(df_anomaly_new))*100, 3)
    #print("False positive rate unique: "+str(false_pos_rate_unique))
    #false_neg_unique = len(df_redteam_sub[df_redteam_sub['real_anomaly_check']==0]['times'])
    #print("Number of false nagetive: "+ str(false_neg_unique))
    #false_neg_rate_unique = round(false_neg_unique/len(df_redteam_sub)*100, 3)
    #print("False negetive rate: "+str(false_neg_rate_unique))
    
    print("Total anomaly point: "+str(sum(df_redteam[df_redteam['time_day']==day-1]['times'])))
else:
    print("No redteam data available!")



