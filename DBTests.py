# Sets type hints to act as string so I can reference a class (in a type hint) before it's defined.
from __future__ import annotations

# For creating a test for a specific column out of a more generic one
import datetime
from functools import partial
# For better typehinting
from typing import List, Callable, Dict

# For graph graphics
import matplotlib
# For getting a dataframe in testing (via read_csv) and setting dataframe to not print dimensions (via options
# attribute). from-import not used to make the purpose of these methods more explicit.
import pandas
# For displaying coverage and result graphs
from matplotlib import pyplot as plt
# For better type hinting, and detecting uses of Series.__getitem__ specifically.
from pandas import DataFrame, Series

from Test import Test, TestResult

matplotlib.use('TkAgg')


class DBTests:
    """
    a testing suite for the data integrity of pandas dataframes.

    this class wraps a pandas dataframe, and can have tests added to to it, simply as
    functions that run on a dataframe row.

    the method :func:`dbtests.run` will iterate over the rows of the database, run the given tests
    and show a coverage report by column — how many of the columns were tested, how many were valid,
    and a sample of invalid rows for each column.
    """

    def __init__(self, df: DataFrame):
        self.dataframe: DataFrame = df
        self.tests: List[Test] = []
        self.columns_tested = set()

    def add_test(self, test_func: Callable[[Series], bool], name: str = None, tested_columns: List[str] = None,
                 ignore_columns: List[str] = None):
        """
        Add a test to the Testing Suite.

        By default, the class will autodetect which columns are being tested as it runs over
        the database. This is fairly crude though, and detects any columns which the test accesses.
        The `tested_columns` and `ignore_columns` parameters can be used to override or modify this behaviour.

        :param test_func: the predicate which will be used to test the rows of the dataframe in this test.

        :param name: a name for the test which will be displayed when running it and can be accessed via the `name`
        property. By default (and if given `None`) this will be set to the name of the predicate function.

        :param tested_columns: setting this parameter to a list of columns will override this behaviour and cause the
        run method to return the given list as the columns tested.

        :param ignore_columns: columns to ignore in tested-columns autodetection. If `tested_columns` is set,
        columns that appear in it and in here will be ignored also.
        """
        test = Test(test_func, name, tested_columns, ignore_columns)
        self.tests.append(test)

    def add_generic_test(self, test_func: Callable[[Series, str], bool], columns: List[str] = None, name: str = None,
                         column_autodetect: bool = False, ignore_columns: List[str] = None):
        """
        Adds a generic test to a group of columns (or all columns). Instead of as in :func:`add_test`, the
        predicate will not only take a row parameter, but also a column parameter preceding it. Individual
        tests will be added for each of the given columns with the predicate column parameter set to them.

        :param test_func: a predicate will be used to test the rows of the dataframe with respect to a given column.

        :param columns: the columns this test should run on. Default is all the columns in the dataframe.

        :param name: a name for the test which will be displayed when running it and can be accessed via the `name`
        property. By default (and if given `None`) this will be set to the name of the predicate function.

        :param column_autodetect: don't set the given column as the tested one, and instead
        autodetect tested columns at runtime. Default false.

        :param ignore_columns: columns to ignore in tested-columns autodetection. Unless column_autodetect is set to
        True, this has no effect
        """
        for column in columns:
            tested_cols = None if column_autodetect else [column]
            self.add_test(partial(test_func, column), name + ' — ' + column, tested_cols, ignore_columns)

    def add_dtype_test(self, test_func: Callable[[Series, str], bool], dtypes: List[str], name: str = None,
                       column_autodetect: bool = False, ignore_columns: List[str] = None):
        """
        Adds a generic test to columns of a certain dtype or dtypes. Instead of as in :func:`add_test`, the
        predicate will not only take a row parameter, but also a column parameter preceding it. Individual
        tests will be added for each of the column of the dtypes given with the predicate column parameter
        set to them.

        :param test_func: a predicate will be used to test the rows of the dataframe with respect to a given column.

        :param name: a name for the test which will be displayed when running it and can be accessed via the `name`
        property. By default (and if given `None`) this will be set to the name of the predicate function.

        :param dtypes: the dtypes this test should run on. Default is all the columns in the dataframe.

        :param column_autodetect: don't set the given column as the tested one, and instead
        autodetect tested columns at runtime. Default false.

        :param ignore_columns: columns to ignore in tested-columns autodetection. Unless column_autodetect is set to
        True, this has no effect
        """
        self.add_generic_test(test_func, self.dataframe.select_dtypes(include=dtypes), name, column_autodetect,
                              ignore_columns)

    def run(self):
        """
        Runs the given tests over the dataframe and returns a matching :class:`DBTestResults` object
        """
        return DBTestResults(
            self.dataframe,
            datetime.datetime.now().strftime('%s'),
            [test.run(self.dataframe) for test in self.tests]
        )


class DBTestResults:
    def __init__(self, dataframe, timestamp, results: List[TestResult]):
        self.dataframe: DataFrame = dataframe
        self.timestamp: int = timestamp
        self.results = results

        # sum([result.columns_tested for result in results])
        self.cols_checked = set().union(*(result.columns_tested for result in results))
        self.validity_by_column: Dict[str, bool] = {}

        for column in self.cols_checked:
            self.validity_by_column[column] = all(result.success for result in self.get_column_results(column))

    def get_column_results(self, column: str) -> List[TestResult]:
        return [result for result in self.results if column in result.columns_tested]

    def print(self, show_valid_cols=False, show_untested=False, stub=False, print_all_failed=False):
        """
        Produces a coverage report of the tests done — how many of the columns were tested, how many were valid,
        and a sample of invalid rows for each column.

        :param show_valid_cols: print result summary for columns that were completely valid. Default false.
        :param show_untested: show columns without tests. Default false.
        :param stub: don't print individual data for each column, just portions tested and valid. Default false.
        :param print_all_failed: print all rows where a test failed. By default only prints up to 10.
        """

        num_rows, num_cols = self.dataframe.shape
        num_checked = len(self.cols_checked)
        num_valid = sum(1 for column, valid in self.validity_by_column.items() if valid)  # Count fully valid columns

        print(f'Columns Tested: {num_checked}/{num_cols} ({round(num_checked / num_cols * 100)}%).')
        print(f'Columns valid: {num_valid}/{num_cols} ({round(num_valid / num_cols * 100, 2)}%).')

        if not stub:  # If stub not set, print details for individual columns.
            print()
            for i, column in enumerate(self.dataframe.columns, 1):
                # Don't show valid or untested columns unless specified.
                if (column in self.cols_checked or show_untested) \
                        and (column in self.cols_checked and not self.validity_by_column[column] or show_valid_cols):
                    print(f'--- Column {i}: {column} ---')
                    for j, result in enumerate(self.get_column_results(column), 1):
                        print(f'Test #{str(j).zfill(2)}: {result.from_test.name}: ', end='')
                        num_failed = len(result.rows_of_failure)
                        num_passed = num_rows - num_failed
                        print(f'{num_passed}/{num_rows} '
                              f'({round(num_passed / num_rows * 100, 2)}%).')

                        pandas.options.display.show_dimensions = False  # Don't show dimensions when printing rows.

                        if print_all_failed:
                            to_print = result.rows_of_failure
                        else:
                            to_print = result.rows_of_failure[:10]

                        for row in to_print:
                            print(row[[self.dataframe.columns[0], column]].to_frame().T.to_string(header=False))
                        if not print_all_failed and len(result.rows_of_failure) > 10:
                            print('...')

                        print()

    def show_summary(self):
        num_rows, num_cols = self.dataframe.shape
        num_checked = len(self.cols_checked)
        num_valid = sum(1 for column, valid in self.validity_by_column.items() if valid)  # Count fully valid columns

        plt.figure(1)
        plt.pie([num_checked, num_cols - num_checked], labels=['Tested', 'Untested'])
        plt.figure(2)
        plt.pie([num_valid, num_cols - num_valid], labels=['Valid', 'Invalid'])
        plt.show()

    def show_column(self):
        pass
