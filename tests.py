from __future__ import annotations

import functools
import sys
from typing import List, Callable, Dict, Tuple, Set

import pandas
from pandas import DataFrame, Series

import unittest


class Test:
    def __init__(self, test: Callable[[Series], bool], name: str = None):
        self.test = test

        if name is None:
            self.name = self.test.__name__
        else:
            self.name = name

    '''
    Returns the tuple of the test result and the columns the test accessed
    '''

    def test_row(self, row: Series) -> Tuple[bool, Set[str]]:
        accessed_columns = set()

        def add_accessed_columns(frame, event, arg):
            if event == 'call' \
                    and frame.f_code == Series.__getitem__.__code__ \
                    and frame.f_locals['self'].equals(row):
                accessed_columns.add(frame.f_locals['key'])

        old = sys.gettrace()
        sys.settrace(add_accessed_columns)

        try:
            test_result = self.test(row)
        finally:
            sys.settrace(old)

        return test_result, accessed_columns

    def run(self, rows: List[Series]) -> TestResult:
        success = True
        columns_tested = set()
        rows_of_failure = []
        for row in rows:
            row_success, columns_tested_in_row = self.test_row(row)
            if not row_success:
                success = False
                rows_of_failure.append(row)
            columns_tested = columns_tested.union(columns_tested_in_row)
        return TestResult(self, columns_tested, success, rows_of_failure)


class TestResult:
    def __init__(self, origin_test: Test, columns_tested: Set[str], success=True, rows_of_failure=None):
        self.from_test = origin_test
        self.columns_tested = columns_tested
        self.success = success
        self.rows_of_failure = [] if self.success else rows_of_failure


class ResultSet:
    def __init__(self, dataframe: DataFrame, column: str, index: int):
        self.dataframe: DataFrame = dataframe
        self.column: str = column
        self.results: List[TestResult] = []
        self.index = index

    def add_result(self, result: TestResult):
        if self.column not in result.columns_tested:
            raise ValueError('Incompatible columns: added test result for columns ' + str(result.columns_tested)
                             + ' for a ' + self.column + ' ResultSet!')

        self.results.append(result)


class DBTests:
    def __init__(self, df: DataFrame):
        self.dataframe: DataFrame = df
        self.tests: List[Test] = []
        self.columns_tested = set()

    def add_test(self, test_func: Callable[[Series], bool], name: str = None):
        try:
            test = Test(test_func, name)
            self.tests.append(test)
        except KeyError:
            raise ValueError('Tried to register a test for a column that does not exist in the dataframe!')

    def run(self, show_valid_cols=False, show_untested=False, stub=False):
        num_rows, num_cols = self.dataframe.shape
        # Tests is a dict by column, so its size is the amount of columns that have tests.
        # Initialize an empty test_result list for each column tested. There will be one result object
        # For each existing test.
        results_by_column: Dict[str, List[TestResult]] = {column: [] for column in self.dataframe}

        rows = [row for i, row in self.dataframe.iterrows()]
        column_validity = {}

        for test in self.tests:
            test_results = test.run(rows)
            for column in test_results.columns_tested:
                results_by_column[column].append(test_results)

        cols_checked = [column for column, result_list in results_by_column.items() if len(result_list) > 0]
        num_cols_checked = len(cols_checked)

        for column in cols_checked:
            column_validity[column] = all(result.success for result in results_by_column[column])

        valid_cols = sum(1 for column, validity in column_validity.items() if validity)
        print(f'Columns Tested: {num_cols_checked}/{num_cols} ({round(num_cols_checked / num_cols * 100)}%).')
        print(f'Columns valid: {valid_cols}/{num_cols} ({round(valid_cols / num_cols * 100, 2)}%).')

        if not stub:
            print()
            for i, column in enumerate(self.dataframe.columns, 1):
                if (column in cols_checked or show_untested) and (not column_validity[column] or show_valid_cols):
                    print(f'--- Column {i}: {column} ---')
                    for j, result in enumerate(results_by_column[column], 1):
                        print(f'Test #{str(j).zfill(2)}: {result.from_test.name}: ', end='')
                        num_failed = len(result.rows_of_failure)
                        num_passed = num_rows - num_failed
                        print(f'{num_passed}/{num_rows} '
                              f'({round(num_passed / num_rows * 100, 2)}%).')

                        pandas.options.display.show_dimensions = False  # Don't show dimensions when printing rows.
                        for row in result.rows_of_failure[:10]:
                            print(row[[self.dataframe.columns[0], column]].to_frame().T.to_string(header=False))
                        if len(result.rows_of_failure) > 10:
                            print('...')

                        print()


if __name__ == '__main__':
    db = pandas.read_csv('College.csv')
    tests = DBTests(db)

    bool_cols = ['Private']

    for bool_col in bool_cols:
        tests.add_test(functools.partial(unittest.boolean_test, bool_col), bool_col + ' Yes/No compliance')

    numerical_cols = ['Apps', 'Accept', 'Enroll', 'Top10perc', 'Top25perc', 'F.Undergrad', 'P.Undergrad', 'Outstate',
                      'Room.Board', 'Books', 'Personal', 'PhD', 'Terminal', 'perc.alumni', 'Expend', 'Grad.Rate']
    for num_col in numerical_cols:
        tests.add_test(functools.partial(unittest.integer_test, num_col), num_col + ' Integer Compliance')

    tests.add_test(unittest.tenpercent_test)

    tests.run()
