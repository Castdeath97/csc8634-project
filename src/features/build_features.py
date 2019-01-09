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
tasks_df = pd.read_csv('data/raw/task-x-y.csv')
"""
pandas.core.frame.DataFrame: Task dataframe for executing task 
"""

def na_per(df):
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

def clean_gpu(gpu_df):
    """Clean gpu dataset by dropping uneeded serial number
    
    Parameters
    ----------
    gpu_df
        gpu dataframe to clean

    Returns
    -------
    pandas.core.frame.DataFrame
        Cleaned GPU dataframe

    """
    gpu_df.drop(columns='gpuSerial', inplace=True)
    return(gpu_df)

def merge_check_task(checkpoints_df, tasks_df):
    """merge checkpoints with task df through job and task id
    
    Parameters
    ----------
    gpu_df
        gpu dataframe to clean

    Returns
    -------
    pandas.core.frame.DataFrame
        Cleaned GPU dataframe

    """   
    print(type(checkpoints_df))
    print(type(tasks_df))

    merged_df = checkpoints_df.merge(tasks_df,
                                     on=['taskId', 'jobId'], how='left')
    return (merged_df)
    
def merge_check_task_gpu(merged_df, gpu_df):
    return(merged_df)
    
na_per(gpu_df)
na_per(checkpoints_df)
na_per(tasks_df)

print(gpu_df.shape)
gpu_df = clean_gpu(gpu_df)
print(gpu_df.shape)

merged_df = merge_check_task(checkpoints_df, tasks_df)
print(merged_df.shape)
print(merged_df.columns)
na_per(merged_df)
