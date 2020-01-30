# Anomaly-Detection-With-Autoencoder

This repository contains the code and some example data used for our paper **Anomaly Detection in Market Data Structures Via Machine Learning Algorithms** which can be found on SSRN:
https://ssrn.com/abstract=3516028 where a detailed discussion of the results can be found.

To run the notebook we use python 3.6.3 (64 bit) the needed packages can be found in *requirementsPython363.txt*.

The main sheet is the jupyter notebook *SwaptionVolatilities.ipynb* which make use of the functions in *helper.py* and *neuralnetwork.py* 
and need the directory structure shown in the repository. Note that you cannot reproduce the pictures shown in the jupyter notebook with the published test data.

An example for the data format for the input swaption volatilities can be found in *input_data* directory.
The complete data set we used are not published. 

The notebook *SwaptionVolatilities_tf2.ipynb* together with the functions in *neuralnetwork_tf2.py* contains an alternative
implementation with make use of the latest tensorflow 2 version together with the keras, which leads to a nicer code.



