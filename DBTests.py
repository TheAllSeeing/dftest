# Sets type hints to act as string so I can reference a class (in a type hint) before it's defined.
from __future__ import annotations

# For creating a test for a specific column out of a more generic one
import datetime
# For compressing generic column, row tests to individual column
from functools import partial
# For running pandasgui in the background (so execution is not blocked until user closes it)
from multiprocessing import Process
# For better typehinting
from typing import List, Callable

# For graph graphics
import matplotlib
# For getting a dataframe in testing (via read_csv) and setting dataframe to not print dimensions (via options
# attribute). from-import not used to make the purpose of these methods more explicit.
import pandas
# For displaying coverage and result graphs
import pandasgui
from matplotlib import pyplot as plt
# For better type hinting, and detecting uses of Series.__getitem__ specifically.
from pandas import DataFrame, Series

import utils
from Test import Test, TestResult
from config import Config, ColumnConfig

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
        self.config = Config()

    def load_config(self, config_file: open):
        """
        Loads a JSON config file.

        A config flle may be something like
        ```
        {
            "Column" : {
                "type": "column_type"
                "tests": [
                    "module.function",
                    "module.class.function"
                ]
                "integrity-levels": {
                  "red": 0,
                  "orange": 0.25,
                  "yellow": 0.5,
                  "blue": 0.75,
                  "green": 1
                }
            }
        }
        ```

        You may specify any of the attributes as you like, and missing values will be infered from the configuration of
        the __DEFAULT__ column if it exists, and otherwise set to type str, no tests and the integrity-levels shown
        above.
        """
        self.config = Config(config_file)

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

    def add_generic_test(self, test_func: Callable[[str, Series], bool], columns: List[str] = None, name: str = None,
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
        dataframe_rows = list(self.dataframe.iterrows())
        base_tests_results = [test.run(dataframe_rows) for test in self.tests]
        config_tests_results = [test.run(dataframe_rows) for test in self.config.get_tests(self.dataframe)]
        return DBTestResults(
            self.dataframe,
            datetime.datetime.now().strftime('%s'),
            base_tests_results + config_tests_results,
            self.config
        )


class ColumnResults:
    """
    The results of tests over a column using one or more multiple predicates.

    This class is used to display results for individual columns, as opposed to thw whole dataframe, both ny writing
    to stdout, displaying summary graphs for the column or individual tests and showing the invalid rows for the column
    in pandasgui.
    """

    def __init__(self, column: str, results: List[TestResult], num_rows: int, config: ColumnConfig):
        """
        :param column: the name of the column
        :param results: a list of each :class:`TestResult` result from the tests ran over the columns.
        :param num_rows: the number of dataframe rows where the column was tested
        """
        self.column = column
        self.results = results
        self.num_rows = num_rows
        self.config = config
        self.invalid_rows = []
        self.figures = []

        # Store a set of the invalid rows. Since rows are not hashable this is not
        # actually a set, and a list of row numbers is instead used to ensure distinction.
        invalid_row_nums = []
        for result in results:
            if not result.success:
                for i, row in result.invalid_rows_tuples():
                    if i not in invalid_row_nums:
                        # Can't check row list directly, since the truth value of a Series is ambiguous.
                        self.invalid_rows.append(row)
                        invalid_row_nums.append(i)

    @property
    def valid(self):
        """
        Whether or not the column completely passes each of the tests ran.
        """
        return all(result.success for result in self.results)

    @property
    def num_tests(self):
        """
        The number of tests ran over the column
        """
        return len(self.results)

    @property
    def num_invalid(self):
        """
        The number of rows where cells in this column failed at least one test.
        """
        return len(self.invalid_rows)

    @property
    def num_valid(self):
        """
        The number of rows where cells in this column passed all tests.
        """
        return self.num_rows - self.num_invalid

    def graph_tests_success(self) -> plt.Figure:
        """
        Generates a stacked bar chart showcasing the number of successes (in green)
        and failures (in red) of each test in the column

        Note: In order for the graphs to show, you have to call `pyplot.show()`

        :returns: the graph's pyplot figure
        """
        labels = [result.from_test.name for result in self.results]
        valids = [result.num_valid for result in self.results]
        invalids = [result.num_invalid for result in self.results]

        fig, axis = plt.subplots()
        axis.bar(labels, valids, label='Valid Rows', color='green')
        axis.bar(labels, invalids, label='Invalid Rows', bottom=valids, color='red')
        axis.legend()
        return fig

    def graph_tests_success_colored(self) -> plt.Figure:
        labels = [result.from_test.name for result in self.results]
        valid_rates = [result.num_valid / self.num_rows for result in self.results]
        colors = [self.config.colorcode(valid_rate) for valid_rate in valid_rates]

        fig, axis = plt.subplots()

        axis.bar(labels, valid_rates, label='Valid Rows', color=colors)
        axis.legend()
        return fig

    def graph_validity(self) -> plt.Figure:
        """
        Generates a pie graph of the column validity as a pyplot figure

        Note: In order for the graphs to show, you have to call `pyplot.show()`

        :returns: the graph's pyplot figure
        """
        fig = plt.figure()
        data = [self.num_valid, self.num_invalid]
        labels = ['Valid', 'Invalid']

        plt.pie(data, autopct=utils.pie_autopct(data), colors=['green', 'red'])
        fig.legend(labels)
        fig.suptitle(self.column + ' Validity')

        return fig

    def open_invalid_rows(self, index):
        """
        Opens the invalid rows at the specified columns in the pandasgui interface.

        :param index: an iterable of the columns to include. This will always include this column.
        """
        index = set(index).union({self.column})
        failures = [result.get_failures(index) for result in self.results]
        pandas_proc = Process(target=pandasgui.show, args=tuple(failures))
        return pandas_proc.start()

    def print(self, columns_to_include=None, column_number=None, print_all_failed=False):
        """
        Prints the results of the column to stdout. Generally, this will generate a title with the column name,
        and print each test, the valid:rows ratio for it and row s where it failed (up to 10)

        :param columns_to_include: columns to include when printing rows of failure.
        By default only row number and this column are printed.

        :param column_number: specify a column index in the printed title.

        :param print_all_failed: print all invalid rows for each test. By default, prints up to 10.
        """
        prefix = f'Column {column_number}: ' if columns_to_include is not None else ''
        print(f'--- {prefix}{self.column} ---')
        for j, result in enumerate(self.results, 1):
            print(f'Test #{str(j).zfill(2)}: {result.from_test.name}: ', end='')
            print(f'{result.num_valid}/{self.num_rows} '
                  f'({round(self.num_valid / self.num_rows * 100, 2)}%).')

            pandas.options.display.show_dimensions = False  # Don't show dimensions when printing rows.

            if print_all_failed:
                to_print = result.invalid_rows
            else:
                to_print = result.invalid_rows[:10]

            for row in to_print:
                columns_to_include = set() if columns_to_include is None else set(columns_to_include)
                columns_to_include = columns_to_include.union({self.column})
                print(row[columns_to_include].to_frame().T.to_string(header=False))
            if not print_all_failed and len(result.invalid_rows) > 10:
                print('...')

            print()


class DBTestResults:
    def __init__(self, dataframe, timestamp, results: List[TestResult], config: Config):
        self.dataframe: DataFrame = dataframe
        self.timestamp: int = timestamp
        self.results = results
        self.config = config

        # self.cols_checked = reduce(set.union, cols_checked)
        self.cols_checked = set().union(*(result.columns_tested for result in results))

    def get_column_results(self, column: str) -> ColumnResults:
        """
        :param column: the column to get the results for
        :return: a :class:`ColumnResults` object containing all the results for tests that tested the specified column
        """
        res_list = [result for result in self.results if column in result.columns_tested]
        return ColumnResults(column, res_list, len(self.dataframe.index), self.config.get_column_config(column))

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
        # Count fully valid columns
        num_valid = sum(1 for column in self.cols_checked if self.get_column_results(column).valid)

        print(f'Columns Tested: {num_checked}/{num_cols} ({round(num_checked / num_cols * 100)}%).')
        print(f'Columns valid: {num_valid}/{num_cols} ({round(num_valid / num_cols * 100, 2)}%).')

        if not stub:  # If stub not set, print details for individual columns.
            print()
            for i, column in enumerate(self.dataframe.columns, 1):
                column_res = self.get_column_results(column)
                # Don't show valid or untested columns unless specified.
                if (column in self.cols_checked or show_untested) \
                        and (column in self.cols_checked and not column_res.valid or show_valid_cols):
                    column_res.print(columns_to_include=[self.dataframe.columns[0]], column_number=i,
                                     print_all_failed=print_all_failed)

    def show_summary(self):
        num_rows, num_cols = self.dataframe.shape
        num_checked = len(self.cols_checked)
        num_valid = sum(1 for column in self.cols_checked if self.get_column_results(column).valid)

        plt.figure(1)
        plt.pie([num_checked, num_cols - num_checked], labels=['Tested', 'Untested'])
        plt.figure(2)
        plt.pie([num_valid, num_cols - num_valid], labels=['Valid', 'Invalid'])
        plt.show()
