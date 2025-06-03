import numpy as np
import pandas as pd

def distances(arr1, arr2):
    """
    Calculates distance values for given arrays.

    Args: 
        arr1 (numpy.ndarray): First array of interest.
        arr2 (numpy.ndarray): Second array of interest.

    Returns:
        (pandas.DataFrame): Renamed pandas.DataFrame of distances.
    """
    return pd.DataFrame(np.linalg.norm(arr1 - arr2, axis=1))


def pull_array(df, label):
    """
    Pulls numpy array of interest from .csv file.

    Args:
        df (pandas.DataFrame): pandas.DataFrame of interest.
        label (Str): Title of column of interest in dataframe.

    Returns:
        (numpy.ndarray): Array of interest.
    """
    
    return np.array(df[[f'{label}_x', f'{label}_y', f'{label}_z']])
    

def avg_stddv(series0, series1, series2, series3):
    """
    Returns mean and standard deviation for given pandas.Series.

    Args:
        series0 (pandas.Series): First series.
        series1 (pandas.Series): Second series.
        series2 (pandas.Series): Third series.
        series3 (pandas.Series): Fourth series.

    Returns: 
        (list): List of means and standard deviations.
    """
    
    return [series0.mean(), series0.std(), series1.mean(), series1.std(), series2.mean(), series2.std(), series3.mean(), series3.std()]
    



