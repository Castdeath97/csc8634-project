"""test_DataPreparation

This python file contains the source code used to test the data preparation
process

Ammar Hasan 150454388 January 2018

"""
import pandas as pd
from src.features import build_features as bf
import pytest

_gpu_df = pd.read_csv('data/raw/gpu.csv')
"""
pandas.core.frame.DataFrame: Global GPU dataframe for gpu stats.
"""
_checkpoints_df = pd.read_csv('data/raw/application-checkpoints.csv')
"""
pandas.core.frame.DataFrame: Global application/event related checkpoints
"""
_task_df = pd.read_csv('data/raw/task-x-y.csv')
"""
pandas.core.frame.DataFrame: Global Task dataframe for executing task 
"""

@pytest.fixture
def global_gpu():
    return(pd.read_csv('data/raw/gpu.csv'))

class TestGPUCleaning(object):

     """ Tests gpu.csv file cleaning   

     """
     def test_serial_drop(self, global_gpu):
        gpu_df = bf.cleanGPU(global_gpu)
        assert not('gpuSerial' in gpu_df.columns)