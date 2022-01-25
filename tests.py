import re
from typing import Any, List, Union

import deepchecks.checks
import numpy as np
from pandas import Series, DataFrame


def range_test(left_limit: Union[int, float] = None, right_limit: Union[int, float] = None,
               left_inclusive=True, right_inclusive=False, cast_as: type = None):
    def test(column, row):
        value = row[column] if cast_as is None else cast_as(row[column])
        return (left_limit is None or left_limit < value) and (right_limit is None or value < right_limit) \
               or left_inclusive and value == left_limit \
               or right_inclusive and value == right_limit

    test.__name__ = f'Column in {"[" if left_inclusive else "("}{left_limit}, {right_limit}{"]" if right_inclusive else ")"}'
    return test


def non_equal_test(value: Any, column=None, name=None):
    if column is None:
        func = lambda column, df: [i for i, cell in enumerate((df[column] != value).values) if cell]
    else:
        func = lambda df: [i for i, cell in enumerate((df[column] != value).values) if cell]
    func.__name__ = f'{"Column" if column is None else column} not {str(value)}' if name is None else name
    return func


def in_list_test(lst: List[Any], column=None):
    if column is None:
        return lambda column, row: row[column] in lst
    return lambda row: row[column] in lst


def match_test(regex: str, column=None, name=None):
    if column is None:
        func = lambda column, df: [i for i, cell in enumerate((df[column].str.match(regex)).values) if cell]
    else:
        func = lambda df: [i for i, cell in enumerate((df[column].str.match(regex)).values) if cell]
    func.__name__ = f'{"Column" if column is None else column} match /{regex}/' if name is None else name
    return func


def simple_type_test(data_types: Union[List[type], type], column=None, name=None):
    data_types = [data_types] if type(data_types) is type else data_types

    if column is None:
        func = lambda column, df: [i for i, cell in enumerate(df[column].apply(lambda x: type(x) not in data_types)) if cell]
    else:
        func = lambda df: [i for i, cell in enumerate(df[column].apply(lambda x: type(x) not in data_types)) if cell]

    type_str = str([data_type.__name__ for data_type in data_types]) if len(data_types) > 1 else data_types[0].__name__
    func.__name__ = f'{"Column" if column is None else column} type {type_str}' if name is None else name
    return func


def is_not_null(column: str, dataframe: DataFrame):
    return [i for i, isnull in enumerate(dataframe[column].isnull().values) if isnull]


is_fraction = range_test(0, 1, True, True)
is_positive = range_test(0, left_inclusive=False)
is_integer = simple_type_test(int)
is_float = simple_type_test(float)
is_str = simple_type_test(str)
