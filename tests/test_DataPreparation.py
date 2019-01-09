"""test_DataPreparation

This python file contains the source code used to test the data preparation
process

Ammar Hasan 150454388 January 2018

"""
import pandas as pd
from src.features import build_features as bf
import pytest

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

merged_df = bf.mergeCheckTask(checkpoints_df, tasks_df) 
"""
pandas.core.frame.DataFrame: Checkpoint and Task merged dataframe
"""

@pytest.fixture
def global_gpu():
    """Fixture used to pass GPU dataset
    
    Returns
    -------
    pandas.core.frame.DataFrame
        GPU dataframe
    """
    return(gpu_df)

   
@pytest.fixture
def global_checkpoints():
    """Fixture used to pass application checkpoint dataset
       
    Returns
    -------
    pandas.core.frame.DataFrame
        application checkpoints dataframe
    """
    return(checkpoints_df)

    
@pytest.fixture
def global_tasks():
    """Fixture used to pass the tasks dataset
    
    Returns
    -------
    pandas.core.frame.DataFrame
        tasks dataframe
    """
    return(tasks_df)
    
@pytest.fixture
def global_merged():
    """Fixture used to pass the application and tasks merged dataframe
    
    Returns
    -------
    pandas.core.frame.DataFrame
        application and tasks merged dataframe
    """
    return(merged_df)

@pytest.mark.usefixtures('global_gpu')
class TestGPUCleaning(object):
     """ Tests gpu.csv file cleaning   

     """
     
     def test_serial_drop(self, global_gpu):      
        """ Tests if serial id was removed

        """
        gpu_df = bf.cleanGPU(global_gpu)
        assert not('gpuSerial' in gpu_df.columns)

@pytest.mark.usefixtures('global_merged')
class testCheckTaskMerge(object):
    """ Tests task and application checkpoints merge

    """
    
    def test_check_col_count(self, global_merged):
        """ Tests if merge has correct number of columns

        """
        assert (global_merged.columns == 9)
    

    def test_check_keys(self, global_merged):             
         """ Tests if keys task and job id are present

         """
         assert (['taskId', 'jobId'] in global_merged.columns)