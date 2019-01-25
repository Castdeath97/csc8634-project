# -*- coding: utf-8 -*-
"""
Introduction
--------------

This python file contains the source code used to test the data preparation
process

Code
------

"""
import pandas as pd
from src.data import make_dataset as md
import pytest
from datetime import datetime

BASE_RAW_DATA_DIR = 'data/raw'
"""
str: Base raw data directory
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

TIMESTAMP_FORMAT = '%Y-%m-%dT%H:%M:%S.%fZ'
"""
str: string used to format timestamp for ms sinc epoch conversion
"""

@pytest.fixture
def global_gpu():
    """Fixture used to pass GPU dataset
    
    Returns
    -------
    pandas.core.frame.DataFrame
        GPU dataframe
    """
    
    GPU_DF = pd.read_csv(GPU_CSV_FILE)
    return(GPU_DF.copy())

   
@pytest.fixture
def global_checkpoints():
    """Fixture used to pass application checkpoint dataset
       
    Returns
    -------
    pandas.core.frame.DataFrame
        application checkpoints dataframe
    """

    CHECK_DF = pd.read_csv(CHECK_CSV_FILE)
    return(CHECK_DF.copy())
  
@pytest.fixture
def global_tasks():
    """Fixture used to pass the tasks dataset
    
    Returns
    -------
    pandas.core.frame.DataFrame
        tasks dataframe
    """
    TASK_DF = pd.read_csv(TASK_CSV_FILE)
    return(TASK_DF.copy())
    
@pytest.fixture
def global_check_task_df():
    """Fixture used to pass the application and tasks merged dataframe
    
    Returns
    -------
    pandas.core.frame.DataFrame
        application and tasks merged dataframe
    """

    CHECK_TASK_DF = md.merge_check_task(pd.read_csv(CHECK_CSV_FILE),
                                        pd.read_csv(TASK_CSV_FILE))
    return(CHECK_TASK_DF.copy())
    
@pytest.fixture
def global_check_task_gpu_df():
    """Fixture used to pass the application, tasks and gpu merged dataframe
    
    Returns
    -------
    pandas.core.frame.DataFrame
        application, tasks and gpu final merged dataframe
    """
    FINAL_MERGED_DF = md.merge_check_task_gpu(
                md.clean_gpu(pd.read_csv(GPU_CSV_FILE)),
                md.clean_check_task(md.merge_check_task(
                        pd.read_csv(CHECK_CSV_FILE),
                        pd.read_csv(TASK_CSV_FILE)
                                   ))
                        )
    return(FINAL_MERGED_DF.copy())    
            
@pytest.fixture
def global_check_task_merge_col_count():
    """Fixture used return expected columns count after check task merge
    
    Returns
    -------
    int
        column count (expected 9)
    """
    return(9)

@pytest.fixture
def global_final_merge_col_count():
    """Fixture used return expected columns count after final merge
    
    Returns
    -------
    int
        column count (expected 12)
    """
    return(12)

@pytest.mark.usefixtures('global_gpu')
class TestGPUCleaning(object):
     """ Tests gpu.csv dataframe cleaning   

     """
     
     def test_serial_drop(self, global_gpu):      
        """ Tests if serial id was removed

        """
        gpu_df = md.clean_gpu(global_gpu)
        assert not('gpuSerial' in gpu_df.columns)
        
     def test_timestamp_conv(self, global_gpu):
        """ Tests if timestamp datetime conversion was done for rows
        
        """     
        assert(md.clean_gpu(global_gpu).timestamp.apply
               (lambda x: isinstance(x, datetime)).all())
        

@pytest.mark.usefixtures('global_check_task_df',
                         'global_check_task_merge_col_count')
class TestCheckTaskMerge(object):
    """ Tests task and application checkpoints merge

    """
    
    def test_check_col_count(self, global_check_task_df, 
                             global_check_task_merge_col_count):
        """ Tests if merge has correct number of columns

        """
        assert (len(global_check_task_df.columns) == 
                global_check_task_merge_col_count)
    
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
        
     def test_timestamp_conv(self, global_check_task_df):
        """ Tests if timestamp datetime conversion was done for rows
        
        """     
        assert(md.clean_check_task(global_check_task_df).timestamp.apply
               (lambda x: isinstance(x, datetime)).all())
        
@pytest.mark.usefixtures('global_check_task_gpu_df',
                         'global_final_merge_col_count')
class TestCheckTaskGPUMerge(object):
    """ Tests task, application checkpoints and gpu final merge
    
    """
    
    def test_check_col_count(self, global_check_task_gpu_df,
                             global_final_merge_col_count):
        """ Tests if merge has correct number of columns
        
        """
        assert (len(global_check_task_gpu_df.columns) ==
                global_final_merge_col_count)
    
    def test_check_keys(self, global_check_task_gpu_df):             
         """ Tests if keys timestamp and hostname are present
         
         """
         cols = ['timestamp', 'hostname']
         assert (global_check_task_gpu_df.columns.isin(cols).any())
         
    def test_nulls(self, global_check_task_gpu_df):
        """ Tests if merge didn't result in any nulls
        
        """
        assert not (global_check_task_gpu_df.isnull().values.any())
        
