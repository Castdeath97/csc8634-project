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

PROCESSED_GPU_CSV_FILE = BASE_PROCESSED_DATA_DIR + '/gpu-processed.csv'
"""
str: gpu-processed.csv final dataset file location 
"""
PROCESSED_CHECK_TASK_CSV_FILE = BASE_PROCESSED_DATA_DIR + '/check-task-processed.csv'
"""
str: check-task-processed.csv final dataset file location 
"""

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
    """Removes uneeded ids for merged application checkpoints and tasks df 

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

    return(check_task_df)

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

    # save final dataset
    gpu_df.to_csv(PROCESSED_GPU_CSV_FILE)
    check_task_df.to_csv(PROCESSED_CHECK_TASK_CSV_FILE)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()
