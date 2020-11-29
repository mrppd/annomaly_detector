#!/usr/bin/python3
"""
Spyder Editor

This is a temporary script file.
"""

import sys
import os
import subprocess
import threading
import time


def worker(num):
    print("Thread created for file: "+str(num))
    subprocess.call('Rscript --vanilla net_test.R '+str(num), shell=True)
    return

total_thread = int(sys.argv[1])
threads = []
for i in range(1,total_thread+1):
    time.sleep(1)
    t = threading.Thread(target=worker, args=(i,))
    threads.append(t)
    t.start()