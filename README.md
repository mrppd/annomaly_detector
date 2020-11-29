<h2>Anomaly detection using unsupervised algorithms </h2>

Presentation slide
https://github.com/mrppd/annomaly_detector/blob/main/Presentation_PronayaProsunDas.pdf

Read me: How to handle the codes? <br/> <br/>

The algorithms have been developed using python 3.6 and R. <br/>
Some of the following important modules are needed to be installed for python, <br/>
1. sklearn <br/>
2. tensorflow <br/>
3. matplotlib <br/>
4. keras <br/>
5. scipy <br/>
6. pylab <br/>

The scripts in flow 2 and 3 have been modified to work with given small dataset and kept in "test_data" folder.

Mainly data preparation is done in R, and model implementation and  classification is performed in python. 

In the python scripts, just change the path according to your folder path and for R, the working directory has to be set as source directory. That's it. 

For flow3, <br/>
    • At first run the “auth_reduction.R” from the “flow3” directory. <br/>
    • It will generate “Test_day_9_10000N200AN.txt_compromised” and “Test_day_9_10000N200AN.txt_reduced” files for the given sample “Test_day_9_10000N200AN.txt”. <br/>
    • Then run “find_pc_reletion.R”. It will generate intermediate file named “src_desTest_day_9_10000N200AN.txt_reduced.csv”. <br/>
    • Then run source_info.R to generate “src_info_src_desTest_day_9_10000N200AN.txt_reduced.csv”. <br/>
    • Finaly run python script “classifier_isolation_forest.py” to get the results. <br/>


For flow2,<br/>
    • At first run the “auth_reduction_v2.R” from “flow2” directory. It will generate “Test_day_9_10000N200AN.txt_compromised” and “Test_day_9_10000N200AN.txt_reduced” files for the given “Test_day_9_10000N200AN.txt”. <br/>
    • Then run python script “classifier_autoencoder.py”. <br/>


For flow1, <br/>
It isn't feasible to run the algorithm of Flow 1 on the given small dataset, as it was designed to work with huge data. <br/>
    • At first,”split.sh” bash script has been used to split the whole authentication data. But for ”split.sh” to work, we need 6 directory named as 1, 2, 3, 4, 5, 6. This part has been done manually for simplicity. Again ”split.sh” has been called from another bash script named “job_split.sh” which is responsible for running the ”split.sh” file in the cluster. After this process we get 59 clunk in each 6 folders, total 354 chunks.  <br/>
    • Then, form those 354 chunks, traffic count (histogram) is calculated using “threader_net_test.py” and kept in the res1, res2, res3, res4, res5, res6 folder. Actually “threader_net_test.py” runs the “net_test.R” in a multi-threaded way.  <br/>
    • After that “merge_res.py” is used to merge all the chunked histograms into one histogram with name “res_combined.csv”.    <br/>      
    • Then we use “generate_service_counts_per_min.R” to find the statistics and saved into a file named “service_per_min2.csv” This will be read by “traffic_anomaly_classifier.py”.   <br/>
    • Finally, using “traffic_anomaly_classifier.py” we trained an auto-encoder and get the classification results. Don’t forget to the path according to your file directory “traffic_anomaly_classifier.py” script.  <br/>
