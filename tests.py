import re
from typing import Any, List

import numpy as np
from pandas import Series, DataFrame


def range_test(test_range: range, left_inclusive=False, right_inclusive=True, cast_as: type = None):
    def test(column, row):
        value = row[column] if cast_as is None else cast_as(row[column])
        return test_range.start < value < test_range.stop \
               or left_inclusive and value == test_range.start \
               or right_inclusive and value == test_range.stop

    test.__name__ = f'Column in {"[" if left_inclusive else "("}{test_range.start}, {test_range.stop}{"]" if right_inclusive else ")"}'
    return test


def nonequal_test(value: Any, column=None, name=None):
    if column is None:
        func = lambda column, df: [i for i, cell in enumerate((df[column] != value).values) if cell]
    else:
        func = lambda df:  [i for i, cell in enumerate((df[column] != value).values) if cell]
    func.__name__ = f'{"Column" if column is None else column} not {str(value)}' if name is None else name
    return func


def in_list_test(lst: List[Any], column=None):
    if column is None:
        return lambda column, row: row[column] in lst
    return lambda row: row[column] in lst


def match_test(regex: str, column=None):
    if column is None:
        return lambda column, row: re.compile(regex)
    return lambda row: re.compile(regex)


def is_not_null(column: str, dataframe: DataFrame):
    return [i for i, isnull in enumerate(dataframe[column].isnull().values) if isnull]


is_fraction = range_test(range(0, 1), left_inclusive=True, right_inclusive=True)


def is_integer(column: str, row: Series):
    return str(row[column]).isdigit()


def is_float(column: str, row: Series):
    try:
        float(row[column])
        return True
    except ValueError:
        return False


def is_positive_integer(column: str, row: Series):
    return row[column].isdigit() and int(row[column]) > 0


def is_positive_float(column: str, row: Series):
    try:
        return float(row[column]) > 0
    except ValueError:
        return False
