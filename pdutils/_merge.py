import warnings
from typing import Union, Optional

import numpy as np
import pandas as pd

def merge_non_na (data: Union[list[Union[pd.Series, np.ndarray, list, tuple, dict]], pd.DataFrame],
                  remaining_na: Optional[str] = 'raise'
                  ) -> pd.Series:
    """
    Combine multiple Series by filling NA values with non-NA values from
    the others.
    
    This function roughly emulates a chain of several pd.Series.combine_first()
    calls. Instead of combining three Series a, b and c as
    ``a.combine_first(b).combine_first(c)`` one can use ``merge_non_na([a,b,c])``.
    
    NA still persists, if the locations of those NA values do not exist or are
    NA as well in all other provided Series.
    
    Parameters
    ----------
    data : sequence of array-like or pd.DataFrame
        Series to merge/choose values from.
    
    remaining_na : {'raise', 'warn', 'ignore'}, default 'raise'
        What to do, when NA values remain after merging:
        - ``raise``: Raise a ValueError.
        - ``warn``: Throw an UserWarning and return resulting Series with
            the remaining NA values.
        - ``ignore``: Return resulting Series with the remaining NA values.
    
    Returns
    -------
    pd.Series
        Result of merging the combined Series.
    
    Notes
    -----
    When providing are mixture of different types of iterables, one has to keep
    in mind that only pd.Series and dicts have an intrinsic index, while the
    index of a list, tuple or ndarray will be numeric. This might lead to
    unexpected results.
    
    Examples
    --------
    >>> s1 = pd.Series([1.0, np.nan])
    >>> s2 = pd.Series([1.0, np.nan])
    >>> s3 = pd.Series([1.0, 2.0])
    >>> merge_non_na([s1,s2,s3])
    0 1.0
    1 2.0
    dtype: float64
    
    NA values still remain, if no non-NA value can be found for that element:
    >>> merge_non_na([s1,s2,s3], remaining_na='ignore')
    0 1.0
    1 np.nan
    dtype: float64
    
    Non-Series objects can be provided as well:
    >>> d1 = {'A': 1.0, 'B': np.nan, 'C': 3.0}
    >>> d2 = {'B': 2.0, 'C': -9.9}
    >>> merge_non_na([d1,d2])
    A 1.0
    B 2.0
    C 3.0
    dtype: float64
    
    With mixed types, care has to be taken about the indices:
    Non-Series objects can be provided as well:
    >>> d1 = {'A': 1.0, 'B': np.nan, 'C': 3.0}
    >>> l2 = [1.0, 2.0, 3.0]
    >>> merge_non_na([d1,l2], remaining_na='ignore')
    A 1.0
    B np.nan
    C 3.0
    0 1.0
    1 2.0
    2 3.0
    dtype: float64
    """
    if isinstance(data, list):
        data = pd.concat([elem if isinstance(elem, pd.Series) else pd.Series(elem) for elem in data], axis='columns')
    
    if not isinstance(data, pd.DataFrame):
        return pd.Series([True for _ in range(len(data))])
    
    n = len(data.columns)
    if n == 0:
        raise ValueError("Number of Series may not be zero.")
    
    i = 0
    result = data.iloc[:,i]
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
        result.where(~na, data.iloc[:,i], inplace=True)
        na = result.isna()
    
    return result
