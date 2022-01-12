# Sets type hints to act as string so I can reference a class (in a type hint) before it's defined.
from __future__ import annotations

# For creating a test for a specific column out of a more generic one
from functools import partial
# For autodetecting accessed columns as a test run, and set the trace function back afterwards.
from sys import settrace, gettrace
# For better typehinting
from typing import List, Callable, Dict, Tuple, Set
# For better type hinting, and detecting uses of Series.__getitem__ specifically.
from pandas import DataFrame, Series

# For getting a dataframe in testing (via read_csv) and setting dataframe to not print dimensions (via options
# attribute). from-import not used to make the purpose of these methods more explicit.
import pandas

# Example testing functions defined by me
import test_funcs


class Test:
    """
    A named predicate to enact on the rows of a :class:`pandas.DataFrame` which test the
    validity of one or more of its columns.
    """

    def __init__(self, test: Callable[[Series], bool], name: str = None, tested_columns: List[str] = None,
                 ignore_columns: List[str] = None):
        """
        :class:`Test` class constructor.

        By default, the class will autodetect which columns are being tested as it runs over
        the database. This is fairly crude though, and detects any columns which the test accesses.
        The `tested_columns` and `ignore_columns` parameters can be used to override or modify this behaviour.

        :param test: the predicate which will be used to test the rows of the dataframe in this test.

        :param name: a name for the test which will be displayed when running it and can be accessed via the `name`
        property. By default (and if given `None`) this will be set to the name of the predicate function.

        :param tested_columns: setting this parameter to a list of columns will override this behaviour and cause the
        run method to return the given list as the columns tested.

        :param ignore_columns: columns to ignore in tested-columns autodetection. If `tested_columns` is set,
        columns that appear in it and in here will be ignored also.
        """
        self.test = test

        if name is None:
            self.name = self.test.__name__
        else:
            self.name = name

        self.tested_columns = set() if tested_columns is None else set(tested_columns)
        self.ignore_columns = set() if ignore_columns is None else set(ignore_columns)

    def test_row(self, row: Series) -> Tuple[bool, Set[str]]:
        """
        :param row: the row to test
        :returns: a tuple of the boolean test result and the set of columns that were tested.
        """

        # If tested_columns is set then no autodetection of tested columns is needed, just test the row
        # and return the defined tested columns (minus ones specified to ignore).
        if len(self.tested_columns) > 0:
            return self.test(row), self.tested_columns.difference(self.ignore_columns)

        # Else, the autodetecting tested columns is needed.
        # This is done via initializing an empty accessed_columns
        # set, then running the test with a tracing function which
        # detects accessed columns and adds them to the set

        accessed_columns = set()

        # More info about sys.settrace and the parameters of trace functions can be found
        # at https://explog.in/notes/settrace.html
        def add_accessed_columns(frame, event, arg):
            # Run on *calls* of *__getitem__* with a *Series* object *which is the test row*
            if event == 'call' \
                    and frame.f_code == Series.__getitem__.__code__ \
                    and frame.f_locals['self'].equals(row):
                accessed_columns.add(frame.f_locals['key'])

        original_trace = gettrace()
        settrace(add_accessed_columns)

        # Make sure to eventually set the trace function back.
        try:
            test_result = self.test(row)
        finally:
            settrace(original_trace)

        return test_result, accessed_columns.difference(self.ignore_columns)

    def run(self, dataframe: DataFrame) -> TestResult:
        """
        Runs the test on a full dataframe.

        :param dataframe: The dataframe to test
        :return: the test result as a TestResult object. the test result as a :class:`TestResult` object.
        """
        columns_tested = set()
        rows_of_failure = []
        for i, row in dataframe.iterrows():
            row_success, columns_tested_in_row = self.test_row(row)
            if not row_success:
                rows_of_failure.append(row)
            columns_tested = columns_tested.union(columns_tested_in_row)
        return TestResult(self, columns_tested, rows_of_failure)


class TestResult:
    """
    The results of a :class:`Test` preformed on a dataframe.
    """

    def __init__(self, origin_test: Test, columns_tested: Set[str], rows_of_failure):
        """
        :param origin_test: the test this is a result of
        :param columns_tested: the columns that were tested on the dataframe (by name)
        :param rows_of_failure: a list of rows where the test failed.
        """
        self.from_test = origin_test
        self.columns_tested = columns_tested
        self.rows_of_failure = rows_of_failure
        self.success = len(rows_of_failure) == 0


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


    def add_generic_test(self, test_func: Callable[[Series, str], bool],  columns: List[str] = None, name: str = None,
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

    def run(self, show_valid_cols=False, show_untested=False, stub=False, print_all_failed=False):
        """
        Iterates over the rows of the database, runs the given tests and shows a coverage report by column — how many
        of the columns were tested, how many were valid, and a sample of invalid rows for each column.

        :param show_valid_cols: print result summary for columns that were completely valid. Default false.
        :param show_untested: show columns without tests. Default false.
        :param stub: don't print individual data for each column, just portions tested and valid. Default false.
        :param print_all_failed: print all rows where a test failed. By default only prints up to 10.
        """
        num_rows, num_cols = self.dataframe.shape
        results_by_column: Dict[str, List[TestResult]] = {column: [] for column in self.dataframe}

        validity_by_column: Dict[str, bool] = {}

        # Run each test over the dataframe and add the results to each of the columns it tested.
        for test in self.tests:
            test_results = test.run(self.dataframe)
            for column in test_results.columns_tested:
                results_by_column[column].append(test_results)

        # Count the columns that have at least one test result (and thus were tested at least once)
        cols_checked = [column for column, result_list in results_by_column.items() if len(result_list) > 0]
        num_cols_checked = len(cols_checked)

        for column in cols_checked:
            validity_by_column[column] = all(result.success for result in results_by_column[column])

        valid_cols = sum(1 for column, valid in validity_by_column.items() if valid)  # Count fully valid columns
        print(f'Columns Tested: {num_cols_checked}/{num_cols} ({round(num_cols_checked / num_cols * 100)}%).')
        print(f'Columns valid: {valid_cols}/{num_cols} ({round(valid_cols / num_cols * 100, 2)}%).')

        if not stub:  # If stub not set, print details for individual columns.
            print()
            for i, column in enumerate(self.dataframe.columns, 1):
                # Don't show valid or untested columns unless specified.
                if (column in cols_checked or show_untested) and (column in cols_checked and not validity_by_column[column] or show_valid_cols):
                    print(f'--- Column {i}: {column} ---')
                    for j, result in enumerate(results_by_column[column], 1):
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


if __name__ == '__main__':
    df = pandas.read_csv('College.csv')  # Example dataset taken from  the book An Introduction to Statistical Learning
    tests = DBTests(df)

    bool_cols = ['Private']
    tests.add_generic_test(test_funcs.boolean_test, name='Yes/No Compliance', columns=bool_cols)

    natural_cols = ['Apps', 'Accept', 'Enroll', 'Top10perc', 'Top25perc', 'F.Undergrad', 'P.Undergrad', 'Outstate',
                      'Room.Board', 'Books', 'Personal', 'PhD', 'Terminal', 'perc.alumni', 'Expend', 'Grad.Rate']
    tests.add_generic_test(test_funcs.nonzero_natural_test, name='Integer Compliance', columns=natural_cols)

    perc_cols = ['Top10perc', 'Top25perc', 'PhD', 'Terminal', 'perc.alumni', 'Grad.Rate']
    tests.add_generic_test(test_funcs.percent_test, name='Percent Compliance', columns=perc_cols)

    tests.add_test(test_funcs.reasonable_room_cost_range_test)
    tests.add_test(test_funcs.apps_accept_enroll_test)
    tests.add_test(test_funcs.sane_spending_test)

    tests.run()
