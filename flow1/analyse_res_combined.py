#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 20:02:55 2019

@author: pronaya
"""


import sys
import os
import subprocess
import threading
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#path = os.path.dirname(os.path.abspath(__file__))
path = "/media/pronaya/My Passport/data science course data"
df_combined = pd.read_csv(path+"/res_combined.csv", sep=',')


83519,983333333
service_count_min = [0] * int(len(df_combined)/60)+1



plt.bar(df_combined['time'],df_combined['hist'])
plt.show()
