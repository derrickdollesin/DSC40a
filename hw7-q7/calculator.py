#!/usr/bin/env python
# coding: utf-8

# <h1><center>Classifying Neutrino Signal from Noise in HPGe Detector</center></h1>

# This notebook will guide you through the second extra credit opportunity. Start out by reading the problem descriptions in the main homework assignment before you work through this notebook.
# 
# We have three imports for this assignment. Please do not import any other packages.

# In[1]:


import numpy as np
import pandas as pd
from itertools import combinations


# In[2]:


waveforms = pd.read_csv('training_classification.csv')
waveforms


# <h2> Problem 6: Prediction Competition<h2>

# In[151]:


def calculate_naive_bayes(curr_amp):
    
    signal_only = waveforms[waveforms["Label"] == 1.0]
    num_signal = signal_only.shape[0]

    #p(class)
    true_positive_rate = num_signal/waveforms.shape[0]
    
    #p(feature|class)
    amps_less_curr_signal = np.count_nonzero(signal_only["Current_Amplitude"] == curr_amp) / num_signal
    
    # p(feature)
    amps_less_curr = np.count_nonzero(waveforms["Current_Amplitude"] >= curr_amp) / waveforms.shape[0]
    
    #p(class|feature)
    return (true_positive_rate) * (amps_less_curr_signal) / (amps_less_curr + 0.000000001) 


# Your task is to modify the `predict` function given below. 
# 

# In[152]:


def predict(row):
    '''Function that returns the predicted score for a given row of the waveform features
    row will be a 1-d array of [tDrift50, tDrift90, tDrift100, blnoise, tslope, Energy, Current_Amplitude]
    please change the return 0 to return a prediction score where:
    * Higher score means the data point is more likely to be a signal (label 1)
    * Lower score means the data point is more likely to be a noise (label 0)
    Note the score doesn't have to be between 0 and 1
    '''
    
    prediction = calculate_naive_bayes(row[6])
    
    return prediction
    


# Don't modify the functions given below. This tests how well your predictions perform on a given dataset.

# In[153]:


def roc_auc(label, score):
    score = np.array(score)
    label = np.array(label)
    dsize = len(score)
    minscore = min(score)
    maxscore = max(score)
    if minscore == maxscore:
        return 0.5
    tpr = []
    fpr = []
    sigscore = score[label==1]
    bkgscore = score[label==0]
    for thr in np.linspace(minscore,maxscore,10000):
        tpr.append(np.sum(sigscore>=thr)/len(sigscore))
        fpr.append(np.sum(bkgscore>=thr)/len(bkgscore))
    
    return np.trapz(tpr,1-np.array(fpr))


# In[154]:


def calculate_AUC(df):
    '''Compute ROC_AUC_score of the predictions corresponding to each row of the given dataframe'''
    n = df.shape[0]
    total_squared_error = 0
    pred_array = []
    label_array = np.array(df.get('Label'))
    for i in np.arange(n):
        pred_array += [predict(df.iloc[i].drop('Label'))]
    return roc_auc(label_array, pred_array)


# You can test out your predictions on the training dataset provided. We'll also test your predictions on a hidden test dataset.

# In[155]:


# An example prediction
example_row = waveforms.iloc[2].drop("Label")
predict(example_row)


# In[156]:


0.8632888309996744
print(calculate_AUC(waveforms))


# <h3> To Submit </h3>
# 
# In the top left corner, in the File menu, select Download as Python (.py). 
# 
# You must save your file as `calculator.py` for the Gradescope autograder to run.

# In[ ]:





# In[ ]:




