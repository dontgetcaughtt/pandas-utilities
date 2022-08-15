from itertools import permutations, pairwise
from typing import Union, Optional

import numpy as np
import pandas as pd

def eq_multiple (series: Union[list[Union[pd.Series, np.ndarray, list, tuple, dict]], pd.DataFrame],
                 na: Optional[str] = 'any'
                 ) -> pd.Series:
    """
    Compare multiple Series elementwise.
    
    This function roughly emulates a chain of several pd.Series.eq() calls.
    Instead of comparing three Series a, b and c as ``a.eq(b) and b.eq(c)``
    one can use ``eq_multiple([a,b,c])``.
    
    Returns True, if only a single element is given.
    
    Parameters
    ----------
    series : sequence of array-like or pd.DataFrame
        List of Series to compare with each other.
    
    na : {'all', 'any, 'none'}, default 'any'
        Control how NA values are handled during the comparison:
        - ``all``: Evaluate to True only, if all elements of
            a given row are NA.
        - ``any``: Ignore NA values and evaluate any comparison
            with NA to True.
        - ``none``: No input Series may contain NA values. Even
            direct comparison of two NA elements evaluates to False.
            Equivalent to ``pd.Series.eq()``.
    
    Returns
    -------
    pd.Series
        Elementwise results of the comparison.
    
    Raises
    ------
    ValuesError
        If ``series`` is empty.
        If elements in ``series`` have different lengths.
        If value for ``na`` is not recognized.
    TypeError
        If elements in ``series`` are not array-like.
    
    Notes
    -----
    Using this function with ``na='none'`` on only two Series is
    equivalent to calling ``pd.Series.eq()`` directly.
    
    See Also
    --------
    equals_multiple: Return True, if all elements are true.
    
    Examples
    --------
    >>> s1 = pd.Series([1.0, 2.0])
    >>> s2 = pd.Series([1.0, np.nan])
    >>> s3 = pd.Series([1.0, np.nan])
    >>> eq_multiple([s1,s2,s3])
    0 True
    1 True
    dtype: bool
    
    Control the evaluation of NA values:
    
    >>> eq_multiple([s1,s2,s3], na='all')
    0 True
    1 False
    dtype: bool
    
    >>> eq_multiple([s1,s2,s3], na='none')
    0 True
    1 False
    dtype: bool
    
    Opposed to the build-in function ``pd.Series.eq()``, eq_multiple
    can consider all-NA rows as True, if provided with ``na='all':
    
    >>> eq_multiple(s2,s3, na='all')
    0 True
    1 True
    dtype: bool
    >>> pd.Series.eq(s2,s3) # or s2.eq(s3)
    0 True
    1 False
    dtype: bool
    
    To check three Series for equality one might easily use two
    built-in ``pd.Series.eq()`` to check if a == b and b == c.
    Yet, if NA values are present in the middle of that chain
    (i.e. b), this might lead to unintended results.
    
    >>> s1 = pd.Series([1.0, 2.0])
    >>> s2 = pd.Series([1.0, np.nan])
    >>> s3 = pd.Series([1.0, 2.0])
    >>> s1.eq(s2) & s2.eq(s3)
    0 True
    1 False
    dtype: bool
    >>> eq_multiple([s1,s2,s3], na='any')
    0 True
    1 True
    dtype: bool
    """
    if isinstance(series, pd.DataFrame):
        series = [s for _, s in series.iteritems()]
    
    if not isinstance(series, list):
        return pd.Series([True for _ in range(len(series))])
    
    if len(series) == 0:
        raise ValueError("List of Series may not be empty.")
    
    n = len(series[0])
    for i, elem in enumerate(series):
        if len(elem) != n:
            raise ValueError("Lengths of all Series must be equal.")
        if isinstance(elem, pd.Series):
            continue
        elif isinstance(elem, (np.ndarray, list, tuple)):
            series.append(pd.Series(series.pop(i)))
        else:
            raise TypeError(f"Element must be Series, ndarray, list or tuple, "
                            "got {type(elem)} instead.")
    
    check = pd.Series([True for _ in range(n)])
    if na == 'any':
        for a, b in permutations(series, r=2):
            check &= pd.Series.eq(a, b) | a.isna() | b.isna()
    elif na == 'all':
        for a, b in pairwise(series):
            check &= pd.Series.eq(a, b) | (a.isna() & b.isna())
    elif na == 'none':
        for a, b in pairwise(series):
            check &= pd.Series.eq(a, b)
    else:
        raise ValueError(f"Expected value for kwarg 'na' to be one of "
                         "['any', 'all', 'none'], got {na} instead.")
    return check

def equals_multiple (series: Union[list[Union[pd.Series, np.ndarray, list, tuple, dict]], pd.DataFrame],
                     na: Optional[str] = 'any'
                     ) -> pd.Series:
    """
    Test whether multiple Series contain the same elements.
    
    This function roughly emulates a chain of several pd.Series.equals() calls.
    Instead of comparing three Series a, b and c as ``a.equals(b) and b.equals(c)``
    one can use ``equals_multiple([a,b,c])``.
    
    Returns True, if only a single element is given.
    
    Parameters
    ----------
    series : sequence of array-like or pd.DataFrame
        List of Series to compare with each other.
    
    na : {'all', 'any, 'none'}, default 'any'
        Control how NA values are handled during the comparison:
        - ``all``: Evaluate to True only, if all elements of
            a given row are NA.
        - ``any``: Ignore NA values and evaluate any comparison
            with NA to True.
        - ``none``: No input Series may contain NA values. Direct
            comparison of two NA elements evaluates to False.
            Equivalent to ``pd.Series.equals()``.
    
    Returns
    -------
    bool
        Result of the comparison.
    
    Raises
    ------
    ValuesError
        If ``series`` is empty.
        If elements in ``series`` have different lengths.
        If value for ``na`` is not recognized.
    TypeError
        If elements in ``series`` are not array-like.
    
    Notes
    -----
    Using this function with ``na='none'`` on only two Series is
    equivalent to calling ``pd.Series.equals()`` directly.
    
    See Also
    --------
    eq_multiple: Elementwise comparison that returns a Series.
    
    Examples
    --------
    >>> s1 = pd.Series([1.0, 2.0])
    >>> s2 = pd.Series([1.0, np.nan])
    >>> s3 = pd.Series([1.0, np.nan])
    >>> equals_multiple([s1,s2,s3])
    True
    
    Control the evaluation of NA values:
    
    >>> equals_multiple([s1,s2,s3], na='all')
    False
    
    >>> equals_multiple([s1,s2,s3], na='none')
    False
    
    Opposed to the build-in function ``pd.Series.equals()``,
    equals_multiple can consider all-NA rows as True, if provided
    with ``na='all':
    
    >>> equals_multiple(s2,s3, na='all')
    True
    >>> pd.Series.eq(s2,s3) # or s2.eq(s3)
    False
    
    To check three Series for equality one might easily use two
    built-in ``pd.Series.equals()`` to check if a == b and b == c.
    Yet, if NA values are present in the middle of that chain
    (i.e. b), this might lead to unintended results.
    
    >>> s1 = pd.Series([1.0, 2.0])
    >>> s2 = pd.Series([1.0, np.nan])
    >>> s3 = pd.Series([1.0, 2.0])
    >>> s1.eq(s2) & s2.eq(s3)
    False
    >>> eq_multiple([s1,s2,s3], na='any')
    True
    """
    return eq_multiple(series, na).all()
