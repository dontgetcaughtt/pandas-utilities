# pandas-utils

![GitHub](https://img.shields.io/github/license/dontgetcaughtt/pandas-utils)
<!-- version, versioning, codecov, status -->

Just a collection of (more or less) useful utility functions for working with pandas.

## Installation

Use pip to install pandas-utils. Note that although all functions already included 
are stable and can be used for production, the API itself is not fixed yet and might 
change in future releases.
```
pip install pandas-utils
```

## Usage
_This section is only meant to give a short overview . For more detailed 
descriptions and additional examples, please refer to the docstrings._

```python
import pdutils
```

### Check multiple Series for equality (or ``NA``)

pandas provides an [``eq()``](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.eq.html) 
method, that compares two Series elementwise. If more than just two Series need 
to be compared, use ``eq_multiple(series, na)``. It returns a Series of boolean 
values. 
```python
>>> s1 = pd.Series([1.0, np.nan, np.nan])
>>> s2 = pd.Series([1.0, np.nan, 3.0])
>>> s3 = pd.Series([np.nan, 2.0, 5.0])
>>> pdutils.eq_multiple([s1,s2,s3])
0 True
1 True
3 False
dtype: bool
```

The build-in pandas method always considers two NA values as unequal (returning 
False). Opposed to that, the keyword argument ``na`` allows to define the behaviour 
when encountering NA values:
- ``all``: Evaluate to True only, if all elements of a given row are NA.
- ``any`` (default): Ignore NA values and evaluate any comparison with NA to True.
- ``none``: No input Series may contain NA values. Even direct comparison of two 
NA elements evaluates to False. Equivalent to ``pd.Series.eq()``.

```python
>>> s1 = pd.Series([1.0, np.nan, np.nan])
>>> s2 = pd.Series([1.0, np.nan, np.nan])
>>> s3 = pd.Series([1.0, 2.0, np.nan])
>>> pdutils.eq_multiple([s1,s2,s3], na='all')
0 True
1 False
2 True
dtype: bool

>>> pdutils.eq_multiple([s1,s2,s3], na='none')
0 True
1 False
2 False
dtype: bool
```

Analogous to the pandas API, ``equals_multiple(series, na)`` returns 
a single boolean value, equivalent to ``eq_multiple().all()``.

### Merge non-NA values

To fill one Series' NA values with non-NA values from another Series, pandas provides a 
[``combine_first()``](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.combine_first.html) 
method. If more than two Series are needed to merge non-NA values, use 
``merge_non_na(data, remaining_na)``:
```python
>>> s1 = pd.Series([1.0, np.nan, np.nan])
>>> s2 = pd.Series([1.0, np.nan, 3.0])
>>> s3 = pd.Series([np.nan, 2.0, 3.0])
>>> pdutils.merge_non_na([s1,s2,s3])
0 1.0
1 2.0
3 3.0
dtype: float64
```

Basically, the function's behaviour is equivalent to a chain of pd.combine_first 
commands:
```python
s1.combine_first(s2).combine_first(s3)
```

``merge_non_NA`` always returns a pd.Series but can take any python iterable as ``data`` 
input. If a pd.DataFrame is provided, all of its columns are considered. The keyword 
argument ``remaining_na`` defines how to deal with NA values remaining after the merge. 
It is possible to throw a ValueError (``raise``, default) or an UserWarning (``warn``), or to 
``ignore`` those cases. Both, warn and ignore, will regularly return a Series with containing 
NA values.

## Contribution

You can contribute in various ways, all of them are highly appreciated:
- **Report bugs** using [GitHub's Issure Tracker](https://github.com/dontgetcaughtt/pandas-utils/issues).
- **Fix bugs** from the list above.
- **Implement additional features** add submit your code through a pull request.
- **Submit feedback**, ideally by filling an [issue](https://github.com/dontgetcaughtt/pandas-utils/issues).

## License

All utilities from this project are licensed under the [MIT license](LICENSE).
