# -*- coding: utf-8 -*-
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from datetime import datetime

# Epoc

TIMESTAMP_FORMAT = '%Y-%m-%dT%H:%M:%S.%fZ'
"""
str: string used to format timestamp for ms sinc epoch conversion
"""

EPOCH = datetime(1970, 1, 1)
"""
datetime: datetime representing epoch used for ms sinc epoch conversion
"""

# Filenames

GPU_CSV_FILE = 'data/raw/gpu.csv'
"""
str: gpu.csv file location 
"""

CHECK_CSV_FILE = 'data/raw/application-checkpoints.csv'
"""
str: application-checkpoints.csv filename file location 
"""

TASK_CSV_FILE = 'data/raw/task-x-y.csv'
"""
str: task-x-y.csv file location 
"""

PROCESSED_CSV_FILE = 'data/processed/processed.csv'
"""
str: processed.csv final dataset file location 
"""

def timestamp_conv(df):
    """ Converts a timestamp to ms since epoch

    Parameters
    ----------
    df
        dataframe to convert its timestamp

    Returns
    -------
    float
         converted timestamp

    """
    df = df.apply(lambda x: 
        (datetime.strptime(x, TIMESTAMP_FORMAT) - EPOCH).total_seconds() * 1000)
    return(df)

def na_per(df):
    """Prints percentage of na values

    Parameters
    ----------
    df
        dataframe to check

    """
    # Count number of na cells and compare to total

    cellCount = np.product(df.shape)
    naCount = df.isna().sum()
    totalNa = naCount.sum()

    # Print NAs values %
    print("The dataframe has: ",
          round(((totalNa/cellCount) * 100), 2), "%", "NAs")

def clean_gpu(gpu_df):
    """Clean gpu dataframe by dropping uneeded serial number and
    fixes timestamp to ms till epoch

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

    # Convert timestamp to ms till epoch

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
    """Removes uneeded ids for merged application checkpoints and tasks df and
    fixes timestamp to ms till epoch

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

    # Convert timestamp format to ms since epoch

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

    # Use fuzzy left join on hostname and timestamp

    keys = ['hostname', 'timestamp']
    
    # set keys as indexes for join 
    
    gpu_df.set_index(keys[1], inplace=True)
    check_task_df.set_index(keys[1], inplace=True)
    
    
    # sort by index
    
    gpu_df.sort_index(inplace=True)
    check_task_df.sort_index(inplace=True)
    
    # Change index to integer   
    gpu_df.index = gpu_df.index.astype(int)
    check_task_df.index = check_task_df.index.astype(int)
 
    
    #check_task_gpu_df = pd.merge_asof(gpu_df, check_task_df, left_index = True,
                                     # right_index = True, tolerance = 20,
                                     # direction = 'nearest')

    check_task_gpu_df = pd.merge_asof(gpu_df, check_task_df, by = keys[0], 
                                      left_index = True, right_index = True,
                                      tolerance = 250, direction = 'nearest')

    check_task_gpu_df.dropna(inplace=True)
    
   # check_task_gpu_df = pd.merge_asof(check_task_df, gpu_df, on = keys[0],
                    #  tolerance = 20, direction = 'nearest' )
    
   # check_task_gpu_df = fz.fuzzy_left_join(check_task_df, gpu_df,
                                         #  left_on = keys, right_on = keys)

   # check_task_gpu_df = check_task_df.merge(gpu_df,
                               #      on=['hostname', 'timestamp'], how='left')
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
    
    clean_gpu_df = clean_gpu(gpu_df)
    merged_df = merge_check_task(checkpoints_df, tasks_df)
    clean_merged_df = clean_check_task(merged_df)
    final_merged_df = merge_check_task_gpu(clean_gpu_df, clean_merged_df)

    # save final dataset
    
    final_merged_df.to_csv(PROCESSED_CSV_FILE)




if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
