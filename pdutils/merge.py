import warnings

import pandas as pd

def merge_non_na_values (data, remaining_na: str = 'raise') -> pd.Series:
    """
    
    Parameters
    ----------
    data : pd.DataFrame or list of pd.Series
        f
    
    remaining_na : {'raise', 'warn', 'ignore'}, default 'raise
    
    
    Returns
    -------
    pd.Series
    
    
    Examples
    --------
    """
    if isinstance(data, list):
        data = pd.concat(data, axis='columns')
    n = len(data.columns)
    
    i = 0
    result = data.iloc[:, i]
    na = result.isna()
    while na.any():
        
        if i + 1 == n:
            if remaining_na == 'raise':
                raise ValueError('NA values still remain in Series after merging.')
            elif remaining_na == 'warn':
                warnings.warn('NA values still remain in Series after merging.')
                return result
            elif remaining_na == 'ignore':
                return result
            else:
                raise ValueError(f"Expected value for kwarg 'remaining_na' to be one of "
                                 "['raise', 'warn', 'ignore'], got {remaining_na} instead.")
        
        i += 1
        result.where(~na, data.iloc[:, i], inplace=True)
        # result = result.combine_first(dataframe[columns[i+1]])
        na = result.isna()
    
    return result
