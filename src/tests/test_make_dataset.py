# -*- coding: utf-8 -*-
"""test_DataPreparation

This python file contains the source code used to test the data preparation
process

Ammar Hasan 150454388 January 2018

"""
import pandas as pd
from src.data import make_dataset as md
import pytest

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

PROCESSED_GPU_CSV_FILE = 'data/processed/gpu-processed.csv'
"""
str: gpu-processed.csv final dataset file location 
"""

PROCESSED_GPU_CSV_FILE = 'data/processed/gpu-processed.csv'
"""
str: gpu-processed.csv final dataset file location 
"""

PROCESSED_CHECK_TASK_CSV_FILE = 'data/processed/check-task-processed.csv'
"""
str: check-task-processed.csv final dataset file location 
"""

@pytest.fixture
def global_gpu():
    """Fixture used to pass GPU dataset
    
    Returns
    -------
    pandas.core.frame.DataFrame
        GPU dataframe
    """
    return(pd.read_csv(GPU_CSV_FILE))

   
@pytest.fixture
def global_checkpoints():
    """Fixture used to pass application checkpoint dataset
       
    Returns
    -------
    pandas.core.frame.DataFrame
        application checkpoints dataframe
    """
    return(pd.read_csv(CHECK_CSV_FILE))

    
@pytest.fixture
def global_tasks():
    """Fixture used to pass the tasks dataset
    
    Returns
    -------
    pandas.core.frame.DataFrame
        tasks dataframe
    """
    return(pd.read_csv(TASK_CSV_FILE))
    
@pytest.fixture
def global_check_task_df():
    """Fixture used to pass the application and tasks merged dataframe
    
    Returns
    -------
    pandas.core.frame.DataFrame
        application and tasks merged dataframe
    """
    return(md.merge_check_task(pd.read_csv(CHECK_CSV_FILE),
                               pd.read_csv(TASK_CSV_FILE)))

@pytest.mark.usefixtures('global_gpu')
class TestGPUCleaning(object):
     """ Tests gpu.csv dataframe cleaning   

     """
     
     def test_serial_drop(self, global_gpu):      
        """ Tests if serial id was removed

        """
        gpu_df = md.clean_gpu(global_gpu)
        assert not('gpuSerial' in gpu_df.columns)


@pytest.mark.usefixtures('global_check_task_df')
class TestCheckTaskMerge(object):
    """ Tests task and application checkpoints merge

    """
    
    def test_check_col_count(self, global_check_task_df):
        """ Tests if merge has correct number of columns

        """
        assert (len(global_check_task_df.columns) == 9)
    

    def test_check_keys(self, global_check_task_df):             
         """ Tests if keys task and job id are present

         """
         cols = ['taskId', 'jobId']
         assert (global_check_task_df.columns.isin(cols).any())
         
@pytest.mark.usefixtures('global_check_task_df')
class TestCheckTaskCleaning(object):
     """ Tests task and application checkpoints merged dataframe cleaning

     """
     
     def test_Id_drop(self, global_check_task_df):      
        """ Tests if task id was removed

        """
        check_task_df = md.clean_check_task(global_check_task_df)        
        cols = ['taskId', 'jobId']
        assert not (check_task_df.columns.isin(cols).any())
        