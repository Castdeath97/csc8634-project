"""bulid_features.py

This python file contains the source code used to clean the dataset to form the 
feature set that will be used to clean the model.

Ammar Hasan 150454388 January 2018

"""

import pandas as pd
import numpy as np

gpu_df = pd.read_csv('data/raw/gpu.csv')
"""
pandas.core.frame.DataFrame: GPU dataframe for gpu stats.
"""
checkpoints_df = pd.read_csv('data/raw/application-checkpoints.csv')
"""
pandas.core.frame.DataFrame: application/event related checkpoints
"""
task_df = pd.read_csv('data/raw/task-x-y.csv')
"""
pandas.core.frame.DataFrame: Task dataframe for executing task 
"""

def naPer(df):
    """Prints percentage of na values
    
    Parameters
    ----------
    df
        dataframe to check

    """

    cellCount = np.product(df.shape)
    naCount = df.isna().sum()
    totalNa = naCount.sum()
    
    # Print NAs values %
    print("The dataframe has: ", 
          round(((totalNa/cellCount) * 100), 2), "%", "NAs")

def cleanGPU(gpu_df):
    gpu_df.drop(columns='gpuSerial', inplace=True)
    return(gpu_df)

def mergeCheckTask(checkpoints_df, task_df):
    return (checkpoints_df)
    
def mergeCheckTaskGPU(merged_df, gpu_df):
    return(merged_df)
    
naPer(gpu_df)
naPer(checkpoints_df)
naPer(task_df)