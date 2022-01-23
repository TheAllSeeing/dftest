from __future__ import annotations

import builtins
import json
import operator
from functools import partial, reduce
from typing import Dict, List, Union, Callable

from pandas import Series, DataFrame

import utils
from Test import Test


class ColumnConfig:
    def __init__(self, column_name: str, data_type: type = None, integrity_levels: Dict[str, int] = None,
                 tests: List[Test] = None):
        self.column_name = column_name
        self.data_type = str if data_type is None else data_type
        if integrity_levels is None:
            self.integrity_levels = {
                                        "red": 0,
                                        "orange": 0.25,
                                        "yellow": 0.5,
                                        "blue": 0.75,
                                        "green": 1
                                    },
        else:
            self.integrity_levels = integrity_levels

        if tests is None:
            self.tests = []
        else:
            self.tests = tests

    def colorcode(self, validity_rate: float):
        color_code = None
        for color in ['red', 'orange', 'yellow', 'blue', 'green']:
            if color not in self.integrity_levels:
                continue
            if validity_rate >= self.integrity_levels[color]:
                color_code = color
            else:
                return color_code
        return 'grey' if color_code is None else color_code  # no valid integrity-level specified


class Config:
    def __init__(self, config_file: open = None):
        if config_file is not None:
            raw_config = json.load(config_file)
            if '__DEFAULT__' in raw_config.keys():
                self.default_dict = raw_config['__DEFAULT__']
                del raw_config['__DEFAULT__']
            else:
                self.default_dict = {}

            self.column_configs = raw_config
        else:
            self.default_dict = {}
            self.column_configs = {}

    def get_default_type(self):
        return self.default_dict.get('type', 'str')

    def get_default_integrity_levels(self):
        return self.default_dict.get(
            'integrity-levels',
            {
                "red": 0,
                "orange": 0.25,
                "yellow": 0.5,
                "blue": 0.75,
                "green": 1
            }
        )

    def get_default_tests(self):
        return self.default_dict.get(
            'tests',
            []
        )

    def _dict_to_column_config(self, column_name: str, column_dict: dict) -> ColumnConfig:

        raw_dtype = column_dict.get('type', self.get_default_type())
        try:
            data_type: type = getattr(builtins, raw_dtype)
        except AttributeError:
            raise ValueError(f'Invalid config for {column_name}: "type" attribute "{raw_dtype}"')

        integrity_levels = column_dict.get('integrity-levels', self.get_default_integrity_levels())

        tests: List[Test] = []
        for test_addr in column_dict.get('tests', self.get_default_tests()):
            try:
                test_func = utils.get_func_from_addr(test_addr)
            except AttributeError:
                raise ValueError(f'Invalid config for {column_name}: invalid test function specified: {test_addr}: '
                                 f'function not found')
            argcount = test_func.__code__.co_argcount

            if argcount == 2:
                test_func = partial(test_func, column_name)
            elif argcount != 1:
                raise ValueError(f'Invalid config for {column_name}: invalid test function specified: {test_addr}: '
                                 f'only (row) or (column, row) params are allowed')

            tests.append(
                Test(test_func, name=f'{test_addr.split(".")[-1]} — {column_name}', tested_columns=[column_name]))

        return ColumnConfig(
            column_name,
            data_type,
            integrity_levels,
            tests
        )

    def get_tests(self, df: DataFrame):
        column_tests_list = [self.get_column_config(column).tests for column in df.columns]
        # Strings together the lists of tests found in column_test_list
        return reduce(operator.concat, column_tests_list)

    def get_column_config(self, column_name: str):
        if column_name in self.column_configs.keys():
            return self._dict_to_column_config(column_name, self.column_configs[column_name])
        else:
            return self._dict_to_column_config(column_name, self.default_dict)
