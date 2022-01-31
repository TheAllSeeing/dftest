# For better type hinting
from typing import Any, List, Union, Iterable
# For dataframe type hinting
from pandas import DataFrame, Series


def from_bool_arr(bool_arr: Iterable[bool]):
    """
    Returns a list of false value indexes from a boolean iterable.

    :param bool_arr: a boolean iterable
    :return: a list of indexes matching false values in the iterable
    """
    return [i for i, cell in enumerate(bool_arr) if not cell]


def in_range_test(left_limit: Union[int, float] = None, right_limit: Union[int, float] = None,
                  left_inclusive=True, right_inclusive=False, preprocess: callable = lambda x: x):
    """
    Creates a generic test function that checks if values for a certain column fall in a given range

    :param left_limit: the lower limit of the range. By default, None, meaning unbounded
    :param right_limit: the upper limit of the range. By default, None, meaning unbounded
    :param left_inclusive: do values equaling the left limit pass the test? By default, false.
    :param right_inclusive: do values equaling the right limit pass the test? By default, false.
    :param preprocess: a function to preprocess values, to ensure comparisons don't raise type errors.
    :return: a test function that checks if values in a dataframe column stay within the specified range
    """
    def test_cell(value):
        return (left_limit is None or left_limit < value) and (right_limit is None or value < right_limit) \
               or left_inclusive and value == left_limit \
               or right_inclusive and value == right_limit

    func = lambda column, df: from_bool_arr(df[column].apply(preprocess).apply(test_cell))
    func.__name__ = f'In range {"[" if left_inclusive else "("}{left_limit}, {right_limit}{"]" if right_inclusive else ")"}'
    return func


def non_equal_test(value: Any):
    """
    Creates a generic test function that checks if values are not equal some given other value.

    :param value: the value to check against
    :return: a generic test function that checks if values are not equal some given other value.
    """
    func = lambda column, df: from_bool_arr((df[column] != value))
    func.__name__ = f'Not {str(value)}'
    return func


def in_list_test(lst: List[Any]):
    """
    Creates a test that checks if values are in a given list

    :param lst: the list to check against
    :return: a generic test function that checks if values in the given column are in the specified list
    """
    func = lambda column, df: from_bool_arr(df[column].apply(lambda x: x in lst))
    func.__name__ = 'In list ' + str(lst)
    return func


def match_test(regex: str):
    """
    :param regex: regular expression to test against
    :return: a generic test function that checks values in the given column match the specified regex
    """
    func = lambda column, df: from_bool_arr(df[column].str.match(regex))
    func.__name__ = f'Match /{regex}/'
    return func


def simple_type_test(data_types: Union[List[type], type]):
    """
    Creates a generic test that checks values are of the specified type.
    Note that this just checks aginst the raw values in the dataframe, which pandas may unexpectedly cast to other types.
    For example, a column of integers with missing values will be automatically converted to floats, to maintain consistency ac
    :param data_types:
    :return:
    """
    data_types = [data_types] if type(data_types) is type else data_types
    type_str = str([data_type.__name__ for data_type in data_types]) if len(data_types) > 1 else data_types[0].__name__

    func = lambda column, df: from_bool_arr(df[column].apply(lambda x: type(x) in data_types))
    func.__name__ = f'Type {type_str}'
    return func


def is_not_null(column: str, dataframe: DataFrame):
    return from_bool_arr(~dataframe[column].isnull())


is_fraction = in_range_test(0, 1, True, True)
is_positive = in_range_test(0, left_inclusive=False)

is_integer = simple_type_test(int)
is_float = simple_type_test(float)
is_str = simple_type_test(str)
