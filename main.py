import re
from typing import Any, List

from pandas import Series


def range_test(test_range: range, left_inclusive=False, right_inclusive=True, cast_as: type = None):
    def test(column, row):
        value = row[column] if cast_as is None else cast_as(row[column])
        return test_range.start < value < test_range.stop \
               or left_inclusive and value == test_range.start \
               or right_inclusive and value == test_range.stop

    return test


def nonequal_test(value: Any, column=None):
    if column is None:
        return lambda column, row: row[column] != value
    return lambda row: row[column] != value


def in_list_test(lst: List[Any], column=None):
    if column is None:
        return lambda column, row: row[column] in lst
    return lambda row: row[column] in lst


def match_test(regex: str, column=None):
    if column is None:
        return lambda column, row: re.compile(regex)
    return lambda row: re.compile(regex)


is_fraction = range_test(range(0, 1), left_inclusive=True, right_inclusive=True)
is_not_null = nonequal_test(None)


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
