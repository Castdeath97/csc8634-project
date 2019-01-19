"""
Introduction
--------------

This python file contains the source code used to carry the data preparation
process

Code
------

"""
# -*- coding: utf-8 -*-
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime

BASE_RAW_DATA_DIR = 'data/raw'
"""
str: Base raw data directory
"""

BASE_PROCESSED_DATA_DIR = 'data/processed'
"""
str: Base processed data directory
"""

GPU_CSV_FILE = BASE_RAW_DATA_DIR + '/gpu.csv'
"""
str: gpu.csv file location 
"""

CHECK_CSV_FILE = BASE_RAW_DATA_DIR +  '/application-checkpoints.csv'
"""
str: application-checkpoints.csv filename file location 
"""

TASK_CSV_FILE = BASE_RAW_DATA_DIR + '/task-x-y.csv'
"""
str: task-x-y.csv file location 
"""

PROCESSED_CSV_FILE = BASE_PROCESSED_DATA_DIR + '/processed.csv'
"""
str: processed.csv final dataset file location 
"""

TIMESTAMP_FORMAT = '%Y-%m-%dT%H:%M:%S.%fZ'
"""
str: string used to format timestamp for datetime conversion
"""

def timestamp_conv(df):
    """ Converts a timestamp to datetime
    
    Parameters
    ----------
    df
        dataframe to convert to datetime
    -------
    float
         converted timestamp
    """
    df = df.apply(lambda x: (datetime.strptime(x, TIMESTAMP_FORMAT)))    
    return(df)

def clean_gpu(gpu_df):
    """Clean gpu dataframe by dropping uneeded serial number and
    fixes timestamp format to datetime

    Parameters
    ----------
    gpu_df
        gpu dataframe to clean

    Returns
    -------
    pandas.core.frame.DataFrame
        Cleaned GPU dataframe

    """

    # Drop uneeded serial column

    gpu_df.drop(columns='gpuSerial', inplace=True)
    gpu_df['timestamp'] = timestamp_conv(gpu_df['timestamp'])
    
    return(gpu_df)

def merge_check_task(checkpoints_df, tasks_df):
    """merge (left join) checkpoints with task df through job and task id

    Parameters
    ----------
    checkpoints_df
        application checkpoints dataframe to merge

    tasks_df
        tasks dataframe to merge

    Returns
    -------
    pandas.core.frame.DataFrame
        Cleaned GPU dataframe

    """

    # Use left join on taskId and jobId

    check_task_df = checkpoints_df.merge(tasks_df,
                                     on=['taskId', 'jobId'], how='left')
    return (check_task_df)

def clean_check_task(check_task_df):
    """Removes uneeded ids and fixes timestamp format to datetime 
    for merged application checkpoints and tasks df

    Parameters
    ----------
    check_task_df
         merged application checkpoints and tasks df to clean

    Returns
    -------
    pandas.core.frame.DataFrame
        Cleaned GPU dataframe

    """

    # Drop uneeded ids

    check_task_df.drop(columns= ['jobId', 'taskId'], inplace=True)
    
    # Fix date format
    
    check_task_df['timestamp'] = timestamp_conv(check_task_df['timestamp'])

    return(check_task_df)

def merge_check_task_gpu(gpu_df, check_task_df):
    """merge (left join) gpu with first merged df through host and timestamp
    
    Parameters
    ----------
    gpu_df
        gpu dataframe to merge
        
    check_task_df
        application checkpoints and tasks megred dataframe to merge with gpu df

    Returns
    -------
    pandas.core.frame.DataFrame
        Cleaned GPU dataframe
    """
    
    # set keys as indexes for join 

    gpu_df.set_index('timestamp', inplace=True)
    check_task_df.set_index('timestamp', inplace=True)
        
    # sort by index
   
    gpu_df.sort_index(inplace=True)
    check_task_df.sort_index(inplace=True)

    # Make timestamp df for first merge 
    
    timestamp_df = check_task_df.copy()
    timestamp_df.drop(['hostname', 'eventName', 
                       'eventType', 'x', 'y', 'level'], axis=1, inplace= True)
    
    # Merge with timestamps only to fix timestamps to nearest in other df ...
    
    gpu_df = pd.merge_asof(gpu_df, timestamp_df,
                           left_index = True, right_index = True,
                           tolerance = pd.Timedelta('0ms'),
                           direction = 'nearest')
    
    # Then merge gpu_df with fixed timestamps with check_task_df
    
    check_task_gpu_df = pd.merge(gpu_df, check_task_df,
                                 on = ['hostname', 'timestamp'])

    return(check_task_gpu_df)

def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    
    # Read datasets in
    
    gpu_df = pd.read_csv(GPU_CSV_FILE)
    checkpoints_df = pd.read_csv(CHECK_CSV_FILE)
    tasks_df = pd.read_csv(TASK_CSV_FILE)
    
    # Cleaning and merging process    
    gpu_df = clean_gpu(gpu_df)
    check_task_df = merge_check_task(checkpoints_df, tasks_df)
    check_task_df = clean_check_task(check_task_df)  
    check_task_gpu_df = merge_check_task_gpu(gpu_df, check_task_df)

    # save final dataset
    
    check_task_gpu_df.to_csv(PROCESSED_CSV_FILE)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    
    project_dir = Path(__file__).resolve().parents[2]

    main()
