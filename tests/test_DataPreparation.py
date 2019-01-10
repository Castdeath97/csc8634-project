"""test_DataPreparation

This python file contains the source code used to test the data preparation
process

Ammar Hasan 150454388 January 2018

"""
import pandas as pd
from src.features import build_features as bf
import pytest

# Read required CSVs

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

check_task_df = bf.merge_check_task(checkpoints_df, tasks_df) 
"""
pandas.core.frame.DataFrame: Checkpoint and Task merged dataframe
"""

check_task_df2 = bf.merge_check_task(checkpoints_df, tasks_df) 
"""
pandas.core.frame.DataFrame: Checkpoint and Task merged dataframe copy for 2nd
merge to tests effecting each other
"""

check_task_gpu_df = bf.merge_check_task_gpu(bf.clean_check_task(check_task_df2)
                                            , gpu_df) 
"""
pandas.core.frame.DataFrame: Checkpoint, Task and GPU merged dataframe
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
def global_check_task_df():
    """Fixture used to pass the application and tasks merged dataframe
    
    Returns
    -------
    pandas.core.frame.DataFrame
        application and tasks merged dataframe
    """
    return(check_task_df)

@pytest.fixture
def global_check_task_gpu_df():
    """Fixture used to pass the application, tasks and gpu merged dataframe
    
    Returns
    -------
    pandas.core.frame.DataFrame
        application, tasks and gpu final merged dataframe
    """
    return(check_task_gpu_df)
    

@pytest.mark.usefixtures('global_gpu')
class TestGPUCleaning(object):
     """ Tests gpu.csv file cleaning   

     """
     
     def test_serial_drop(self, global_gpu):      
        """ Tests if serial id was removed

        """
        gpu_df = bf.clean_gpu(global_gpu)
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
        check_task_df = bf.clean_check_task(global_check_task_df)        
        cols = ['taskId', 'jobId']
        assert not (check_task_df.columns.isin(cols).any())
        

@pytest.mark.usefixtures('global_check_task_gpu_df')
class TestCheckTaskGPUMerge(object):
    """ Tests task, application checkpoints and gpu final merge

    """
    
    def test_check_col_count(self, global_check_task_gpu_df):
        """ Tests if merge has correct number of columns

        """
        assert (len(global_check_task_gpu_df.columns) == 13)
    

    def test_check_keys(self, global_check_task_gpu_df):             
         """ Tests if keys timestamp and hostname are present

         """
         cols = ['timestamp', 'hostname']
         assert (global_check_task_gpu_df.columns.isin(cols).any())