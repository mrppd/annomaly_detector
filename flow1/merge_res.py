#!/usr/bin/python3
"""
Created on Wed Jul  3 15:08:04 2019

@author: pronaya
"""

import sys
import os
import subprocess
import threading
import time
import pandas as pd
import numpy as np


path = os.path.dirname(os.path.abspath(__file__))
#path = "/media/pronaya/My Passport/data science course data"
total_dir = 2
files_per_dir = 59

df_1 = pd.read_csv(path+"/res1/1.csv", sep=',')

for i in range(1, total_dir+1):
    for j in range(1, files_per_dir+1):
        if(i==1 and j==1):
            continue
        file_name = path+"/res"+str(i)+"/"+str(j)+".csv"
        df_2 = pd.read_csv(file_name, sep=',')
        
        df_1['hist'] = df_1['hist']+df_2['hist']
        print("File "+str(j)+".csv in directory "+str(i)+" has been processed!")
        
df_1.to_csv("res_combined.csv", sep=',', index=False) 