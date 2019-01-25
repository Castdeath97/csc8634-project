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
import sqlite3

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
    """merge (left join) gpu df with first merged df through host and timestamp
    
    Parameters
    ----------
    check_task_df
        application checkpoints and tasks megred dataframe to merge with gpu df
    
    gpu_df
        gpu dataframe to merge

    Returns
    -------
    pandas.core.frame.DataFrame
        Cleaned GPU dataframe
    """
    
    # Record start and stop times for events and drop old timestamps

    check_task_df_start = check_task_df[
    check_task_df['eventType'] == 'START']
    check_task_df_stop = check_task_df[
    check_task_df['eventType'] == 'STOP']

    check_task_df_start.rename(
            index=str, columns={"timestamp": "start_time"}, inplace = True)
    check_task_df_stop.rename(
            index=str, columns={"timestamp": "stop_time"}, inplace = True)

    check_task_df_stop.drop('eventType', axis = 1, inplace = True)
    check_task_df_start.drop('eventType', axis = 1, inplace = True)
   
    # Make each field record start and stop combined
   
    check_task_df = pd.merge( check_task_df_start, check_task_df_stop, 
                on=['hostname', 'eventName', 'x', 'y', 'level'])
   
    # Remove any timestamps that occur out of the gpu dataset
   
    check_task_df = check_task_df[
            (check_task_df['start_time'] >= gpu_df['timestamp'][0]) &
            (check_task_df['stop_time']
            <= gpu_df['timestamp'][len(gpu_df)-1])]
   
    # Use sqllite to only combine with gpu if timestamp is between times

    # connection to sql
    conn = sqlite3.connect(':memory:')

    # move dataframes to sql
    check_task_df.to_sql('CheckTask', conn, index=False)
    gpu_df.to_sql('Gpu', conn, index=False)

    # SQL query
    query = '''
    SELECT *
    FROM Gpu
    LEFT JOIN CheckTask ON gpu.hostname = CheckTask.hostname
    WHERE gpu.timestamp >= CheckTask.start_time 
        AND gpu.timestamp <= CheckTask.stop_time
    '''
    # get new df
    merged_df = pd.read_sql_query(query, conn)
    
    # drop duplicate hostname row (index 8)
    merged_df = merged_df.loc[:,~merged_df.columns.duplicated()]
    
    # group for averages (average stats for every task)
    
    functions = {
        'powerDrawWatt': 'mean', 'gpuTempC': 'mean',
        'gpuUtilPerc': 'mean', 'gpuMemUtilPerc': 'mean',
        'start_time': 'first', 'stop_time': 'first', 
        'gpuUUID' : 'first'}
    
    merged_df = merged_df.groupby(
        ['hostname', 'eventName', 'x', 'y', 'level'],
        as_index=False, sort=False
    ).agg(functions)

    return(merged_df)

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
