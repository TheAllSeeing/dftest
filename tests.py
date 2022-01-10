from __future__ import annotations

import functools
from typing import List, Callable, Dict

import pandas
from pandas import DataFrame, Series

import columndetection
import unittest


class Test:
    def __init__(self, column: str, test: Callable[[Series], bool], name: str = None):
        self.column = column
        self.test = test

        if name is None:
            self.name = self.test.__name__
        else:
            self.name = name

    def run(self, rows: List[Series]) -> TestResult:
        success = True
        rows_of_failure = []
        for row in rows:
            if not self.test(row):
                success = False
                rows_of_failure.append(row)
        return TestResult(self, success, rows_of_failure)


class TestResult:
    def __init__(self, from_test: Test, success=True, rows_of_failure=None):
        self.from_test = from_test
        self.success = success
        self.rows_of_failure = [] if self.success else rows_of_failure


class ResultSet:
    def __init__(self, dataframe: DataFrame, column: str):
        self.dataframe: DataFrame = dataframe
        self.column: str = column
        self.results: List[TestResult] = []

    def add_result(self, result: TestResult):
        if result.from_test.column != self.column:
            raise ValueError('Incompatible columns: added test result for column ' + result.from_test.column
                             + ' for a ' + self.column + ' ResultSet!')

        self.results.append(result)


class DBTests:
    def __init__(self, df: DataFrame):
        self.dataframe: DataFrame = df
        self.tests: Dict[str, List[Test]] = {}
        for column in df:
            self.tests[column] = []

    def add_test(self, column: str, test_func: Callable[[Series], bool], name: str = None):
        try:
            test = Test(column, test_func, name)
            self.tests[test.column].append(test)
            return True
        except KeyError:
            raise ValueError('Tried to register a test for a column that does not exist in the dataframe!')

    def run(self, show_valid_cols=False, show_untested=False, stub=False):

        num_rows, num_cols = self.dataframe.shape
        # Tests is a dict by column, so its size is the amount of columns that have tests.
        cols_checked = [column for column, column_tests in self.tests.items() if len(column_tests) > 0]
        num_cols_checked = len(cols_checked)
        # Initialize an empty test_result list for each column tested. There will be one result object
        # For each existing test.
        results_by_column: Dict[str, List[TestResult]] = {column: [] for column in self.tests.keys()}

        rows = [row for i, row in self.dataframe.iterrows()]

        column_validity = {}
        for column in results_by_column.keys():
            for test in self.tests[column]:
                results_by_column[column].append(test.run(rows))  # Run it on the cell and

            column_validity[column] = all(result.success for result in results_by_column[column])

        valid_cols = sum(1 for column, validity in column_validity.items() if validity)
        print(f'Columns Tested: {num_cols_checked}/{num_cols} ({round(num_cols_checked / num_cols * 100)}%).')
        print(f'Columns valid: {valid_cols}/{num_cols} ({round(valid_cols/num_cols*100, 2)}%).')

        if not stub:
            print()
            for i, column in enumerate(self.tests.keys(), 1):
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
        tests.add_test(bool_col, functools.partial(unittest.boolean_test, bool_col), bool_col + ' Yes/No compliance')

    numerical_cols = ['Apps', 'Accept', 'Enroll', 'Top10perc', 'Top25perc', 'F.Undergrad', 'P.Undergrad', 'Outstate',
                      'Room.Board', 'Books', 'Personal', 'PhD', 'Terminal', 'perc.alumni', 'Expend', 'Grad.Rate']
    for num_col in numerical_cols:
        tests.add_test(num_col, functools.partial(unittest.integer_test, num_col), num_col + ' Integer Compliance')

    tests.add_test('Top10perc', unittest.tenpercent_test)


    def my_test():
        unittest.tenpercent_test(db.iloc[1])


    # columndetection.get_affecting_columns(unittest.tenpercent_test)
    # print(columndetection.test_call(my_test, db.__getitem__))
    tests.run()
