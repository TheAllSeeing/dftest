# For better type hinting
from typing import Any, List, Union
# For dataframe type hinting
from pandas import DataFrame


def in_range_test(left_limit: Union[int, float] = None, right_limit: Union[int, float] = None,
                  left_inclusive=True, right_inclusive=False, name=None):
    def test_cell(value):
        return (left_limit is None or left_limit < value) and (right_limit is None or value < right_limit) \
               or left_inclusive and value == left_limit \
               or right_inclusive and value == right_limit

    func = lambda column, df: [i for i, cell in df[column].apply(test_cell) if not cell]
    func.__name__ = f'In range {"[" if left_inclusive else "("}{left_limit}, {right_limit}{"]" if right_inclusive else ")"}' if name is None else name
    return func


def non_equal_test(value: Any, name=None):
    func = lambda column, df: [i for i, cell in enumerate((df[column] != value).values) if cell]
    func.__name__ = f'Not {str(value)}' if name is None else name
    return func


def in_list_test(lst: List[Any], name=None):
    func = lambda column, df: [i for i, cell in enumerate(df[column].apply(lambda x: x in lst)) if not cell]
    func.__name__ = 'In list ' + str(lst) if name is None else name
    return func


def match_test(regex: str, name=None):
    func = lambda column, df: [i for i, cell in enumerate((df[column].str.match(regex)).values) if not cell]
    func.__name__ = f'Match /{regex}/' if name is None else name
    return func


def simple_type_test(data_types: Union[List[type], type], name=None):
    data_types = [data_types] if type(data_types) is type else data_types
    type_str = str([data_type.__name__ for data_type in data_types]) if len(data_types) > 1 else data_types[0].__name__

    func = lambda column, df: [i for i, cell in enumerate(df[column].apply(lambda x: type(x) not in data_types)) if cell]
    func.__name__ = f'Type {type_str}' if name is None else name
    return func


def is_not_null(column: str, dataframe: DataFrame):
    return [i for i, isnull in enumerate(dataframe[column].isnull().values) if isnull]


is_fraction = in_range_test(0, 1, True, True)
is_positive = in_range_test(0, left_inclusive=False)

is_integer = simple_type_test(int)
is_float = simple_type_test(float)
is_str = simple_type_test(str)
