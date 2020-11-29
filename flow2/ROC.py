#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 00:56:35 2019

@author: pronaya
"""

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

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

path = "/media/pronaya/My Passport1/data science course data/test_data/flow2"
df_roc = pd.read_csv(path+"/roc_data.txt", sep=";")
df_roc['tpr'] = df_roc['tpr']/100
df_roc['fpr'] = df_roc['fpr']/100

# This is the AUC
auc = np.trapz(df_roc['tpr'], df_roc['fpr'])*1

# This is the ROC curve
plt.title('Receiver Operating Characteristic for flow 2')
plt.plot(df_roc['fpr'],  df_roc['tpr'], 'b', label = 'AUC = %0.2f' % auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1.03])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig(path+'/flow2.png', format='png', dpi=600)