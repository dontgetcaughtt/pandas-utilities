from itertools import permutations, pairwise
import warnings

import pandas as pd


def eq_series (series, na='any'):
    check = pd.Series([True for _ in range(len(series[0]))])
    
    # for a, b in permutations(series, r=2):
    #     eq = pd.Series.eq(a,b)
    #     if na == 'any':
    #         eq |= a.isna() | b.isna()
    #     elif na == 'all':
    #         eq |= a.isna() & b.isna()
    #     elif na == 'none':
    #         pass
    #     else:
    #         raise ValueError(f'Unknown keyword argument {na}.')
    #     check &= eq
    
    if na == 'any':
        for a, b in permutations(series, r=2):
            check &= pd.Series.eq(a ,b) | a.isna() | b.isna()
    elif na == 'all':
        for a, b in pairwise(series):
            check &= pd.Series.eq(a ,b) | (a.isna() & b.isna())
    elif na == 'none':
        for a, b in pairwise(series):
            check &= pd.Series.eq(a ,b)
    else:
        raise ValueError(f'Unknown keyword argument {na}.')
    
    return check

def equal_series (series, na='any'):
    return eq_series(series, na).all()

def merge_non_na_values (data, remaining_na='raise'):
    
    if isinstance(data, list):
        data = pd.concat(data, axis='columns')
    
    n = len(data.columns)
    
    i = 0
    result = data.iloc[:,i]
    na = result.isna()
    while na.any():
        
        if i + 1 == n:
            match remaining_na:
                case 'raise':
                    raise ValueError('NA values still remain in Series after merging.')
                case 'warn':
                    warnings.warn('NA values still remain in Series after merging.')
                    return result
                case 'ignore':
                    return result
                case _:
                    raise ValueError(f"Expected value for kwarg 'remaining_na' to be in "
                                     "['raise', 'warn', 'ignore']. Got {remaining_na} instead.")
        
        i += 1
        result.where(~na, data.iloc[:,i], inplace=True)
        # result = result.combine_first(dataframe[columns[i+1]])
        na = result.isna()
    
    return result
