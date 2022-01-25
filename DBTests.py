# Sets type hints to act as string so I can reference a class (in a type hint) before it's defined.
from __future__ import annotations

# For creating a test for a specific column out of a more generic one
import datetime
# For compressing generic column, row tests to individual column
import operator
import re
from configparser import ConfigParser
from functools import partial, reduce
# For running pandasgui in the background (so execution is not blocked until user closes it)
from multiprocessing import Process
# For better typehinting
from typing import List, Callable, Hashable, Iterable

# For graph graphics
import matplotlib
# For getting a dataframe in testing (via read_csv) and setting dataframe to not print dimensions (via options
# attribute). from-import not used to make the purpose of these methods more explicit.
import numpy as np
import pandas
# For displaying coverage and result graphs
import pandasgui
import seaborn
from matplotlib import pyplot as plt
# For better type hinting, and detecting uses of Series.__getitem__ specifically.
from pandas import DataFrame, Index

import utils
from Test import TestResult, Test
from style import StyleFile, Style

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

    def load_config(self, config_file: str):
        """
        Loads a JSON config file.

        ```
        [testmodule.test_func]
        name = mytestbane
        include = col1_to_test,col2_to_test
        exclude = col1_to_ignore,col2_to_ignore
        autodetect = true
        ignore = false
        ```
        any tests specified in brackets like above will be added. all of the settings below it are optional. Generic vs.
        normal tests are detected autpmatically

        for generic tests, include and exclude determine which columns will be tested, and autodetect toggle autodetection.

        for normal tests, include overrides column autodetection and exclude specifies columns to ignore in
        autodetection (or in reading include). "autodetection" value has no effect.

        set ignore to true in order to easily disable tests. name controls test name.
        """
        config = ConfigParser()
        config.read(config_file)
        for test in config.sections():
            test_cfg = config[test]

            if not test_cfg.getboolean('ignore', False):
                try:
                    test_func = utils.get_func_from_addr(test)
                except ValueError as e:
                    raise ValueError(f'Nonexistent test specified in {config_file}: {test}.')

                argcount = test_func.__code__.co_argcount
                included = test_cfg.get('include', None)
                # Use regex to allow escaping commas
                included = re.split(r'(?<!\\),', included) if included is not None else included

                excluded = test_cfg.get('exclude', None)
                # Use regex to allow escaping commas
                excluded = re.split(r'(?<!\\),', excluded) if excluded is not None else excluded

                name = test_cfg.get('name', None)
                if argcount == 2:
                    self.add_generic_test(test_func, included,  name, test_cfg.getboolean('autodetect', False), excluded)
                elif argcount == 1:
                    self.add_test(test_func, name, included, excluded)
                else:
                    raise ValueError(f'Invalid test specified: {test}: function argcount {argcount};'
                                     f'only (row) or (column, row) params are allowed')


    def add_test(self, test_func: Callable[[DataFrame], List[Hashable]], name: str = None,
                 tested_columns: List[str] = None, ignore_columns: List[str] = None):
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
        self.tests.append(Test(test_func, self.dataframe.columns, name, tested_columns, ignore_columns))

    def add_generic_test(self, test_func:  Callable[[DataFrame], List[Hashable]], columns: Iterable[str] = None,
                         name: str = None, column_autodetect: bool = False, ignore_columns: List[str] = None):
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
        for column in (self.dataframe.columns if columns is None else columns):
            tested_cols = None if column_autodetect else [column]
            func_name = test_func.__name__ if name is None else name
            self.add_test(partial(test_func, column), func_name + ' — ' + column, tested_cols, ignore_columns)

    def clear(self):
        """
        Removes all added tests
        """
        self.tests.clear()

    def run(self):
        """
        Runs the given tests over the dataframe and returns a matching :class:`DBTestResults` object
        """
        results = []
        for i, test in enumerate(self.tests):
            print(f'\rTesting {round(i/len(self.tests)*100):02d}% (#{i+1}: {test.name})', end='')
            results.append(test.run(self.dataframe))
        print('\rFinished testing')

        return DBTestResults(
            self.dataframe,
            datetime.datetime.now().strftime('%s'),
            results
        )


class ColumnResults:
    """
    The results of tests over a column using one or more multiple predicates.

    This class is used to display results for individual columns, as opposed to thw whole dataframe, both ny writing
    to stdout, displaying summary graphs for the column or individual tests and showing the invalid rows for the column
    in pandasgui.
    """

    def __init__(self, column: str, results: List[TestResult], dataframe: DataFrame, style: Style):
        """
        :param column: the name of the column
        :param results: a list of each :class:`TestResult` result from the tests ran over the columns.
        :param dataframe: the tested dataframe
        """
        self.column = column
        self.results = results
        self.style = style
        self.dataframe = dataframe

        self.style = style
        try:
            self.invalid_row_index = set(reduce(operator.concat, [result.invalid_row_index for result in results]))
        except TypeError:  # Reduce throws a TypeError if it's given an empty list.
            self.invalid_row_index = set()

    @property
    def tested(self):
        return len(self.results) > 0

    @property
    def valid(self):
        """
        Whether the column completely passes each of the tests ran.
        """
        return all(result.success for result in self.results)

    @property
    def num_tests(self):
        """
        The number of tests ran over the column
        """
        return len(self.results)

    @property
    def num_rows(self):
        return len(self.dataframe.index)

    @property
    def num_invalid(self):
        """
        The number of rows where cells in this column failed at least one test.
        """
        return len(self.invalid_row_index)

    @property
    def num_valid(self):
        """
        The number of rows where cells in this column passed all tests.
        """
        return self.num_rows - self.num_invalid

    def load_stylefile(self, filepath):
        self.stylefile = StyleFile(filepath)

    def graph_tests_success(self) -> plt.Figure:
        """
        Generates a stacked bar chart showcasing the number of successes (in green)
        and failures (in red) of each test in the column

        Note: In order for the graphs to show, you have to call `pyplot.show()`

        :returns: the graph's pyplot figure
        """
        labels = [result.from_test.name for result in self.results]
        valid_nums = [result.num_valid for result in self.results]
        valid_colors = [self.style.colorcode(valid_num / self.num_rows) for valid_num in valid_nums]
        invalid_nums = [result.num_invalid for result in self.results]
        invalid_colors = [utils.adjust_lightness(valid_color, 0.4) for valid_color in valid_colors]

        fig, axis = plt.subplots()
        axis.bar(labels, valid_nums, label='Valid Rows', color=valid_colors)
        axis.bar(labels, invalid_nums, label='Invalid Rows', bottom=valid_nums, color=invalid_colors)
        axis.legend()
        return fig

    def graph_validity_heatmap(self):
        test_labels = [result.from_test.name for result in self.results]
        data = np.array([result.num_valid / self.num_rows for result in self.results])
        fig, ax = plt.subplots()

        step_colors, step_values = self.style.transposed
        color_map = utils.nonlinear_cmap(step_colors, step_values)

        seaborn.heatmap([data],
                        vmin=0, vmax=1,
                        square=True,
                        cmap=color_map,
                        cbar_kws=dict(use_gridspec=False, location="bottom"),
                        annot=True, fmt='.1%',
                        xticklabels=test_labels, yticklabels=False)

        for t in ax.texts: t.set_text(t.get_text() + " %")

    def graph_validity(self) -> plt.Figure:
        """
        Generates a pie graph of the column validity as a pyplot figure

        Note: In order for the graphs to show, you have to call `pyplot.show()`

        :returns: the graph's pyplot figure
        """
        fig = plt.figure()
        data = [self.num_valid, self.num_invalid]
        labels = ['Valid', 'Invalid']
        colors = [self.style.values[-1][0], self.style.values[0][0]]

        plt.pie(data, autopct=utils.make_autopct(data), colors=colors)
        fig.legend(labels)
        fig.suptitle(self.column + ' Validity')
        return fig

    def open_invalid_rows(self, index, sample_size: int = None):
        """
        Opens the invalid rows at the specified columns in the pandasgui interface.

        :param index: an iterable of the columns to include. This will always include this column.
        :param sample_size: if specified, opens the first n invalid rows.
        """
        sample_size = self.num_invalid if sample_size is None else sample_size
        index = set(index).union({self.column})
        failures = [result.get_invalid_rows(self.dataframe)[index].iloc[:sample_size] for result in self.results]
        pandas_proc = Process(target=pandasgui.show, args=tuple(failures))
        return pandas_proc.start()

    def print(self, columns_to_include=None, column_number=None, print_all_failed=False):
        """
        Prints the results of the column to stdout. Generally, this will generate a title with the column name,
        and print each test, the valid:rows ratio for it and row s where it failed (up to 10)

        :param columns_to_include: columns to include when printing rows of failure.
        By default, only row number and this column are printed.

        :param column_number: specify a column index in the printed title.

        :param print_all_failed: print all invalid rows for each test. By default, prints up to 10.
        """
        prefix = f'Column {column_number}: ' if columns_to_include is not None else ''
        print(f'--- {prefix}{self.column} ---')
        for j, result in enumerate(self.results, 1):
            print(f'Test #{str(j).zfill(2)}: {result.from_test.name}: ', end='')
            print(f'{result.num_valid}/{self.num_rows} '
                  f'({round(result.num_valid / self.num_rows * 100, 2)}%).')

            pandas.options.display.show_dimensions = False  # Don't show dimensions when printing rows.

            if print_all_failed:
                to_print = result.get_invalid_rows(self.dataframe)
            else:
                to_print = result.get_invalid_rows(self.dataframe).iloc[:10]

            columns_to_include = set() if columns_to_include is None else set(columns_to_include)
            columns_to_include = columns_to_include.union({self.column})
            if not len(to_print.index) == 0:
                print(to_print[columns_to_include])
            if not print_all_failed and result.num_invalid > 10:
                print('...')

            print()


class DBTestResults:
    plt = plt

    def __init__(self, dataframe, timestamp, results: List[TestResult]):
        self.dataframe: DataFrame = dataframe
        self.timestamp: int = timestamp
        self.results = results

        self.cols_checked = set().union(*(result.columns_tested for result in results))
        self.invalid_row_index = set().union(*(result.invalid_row_index for result in results))

        self.stylefile = StyleFile()

    @property
    def num_rows(self):
        """Number of rows tested"""
        return len(self.dataframe.index)

    @property
    def num_cols_tested(self):
        """Number of columns tested"""
        return len(self.cols_checked)

    @property
    def num_cols_untested(self):
        return len(self.dataframe.columns) - self.num_cols_tested

    @property
    def num_cols_valid(self):
        return sum(1 for column in self.column_results if column.tested and column.valid)

    @property
    def num_cols_invalid(self):
        return self.num_cols_tested - self.num_cols_valid

    @property
    def num_rows_invalid(self):
        """Number of rows that came as invalid under at least one test"""
        return len(self.invalid_row_index)

    @property
    def num_rows_valid(self):
        """Number of rows that passes all tests ran"""
        return self.num_rows - self.num_rows_invalid

    @property
    def column_results(self):
        """The list of :class:`ColumnResults` objects for each column in the tested dataframe"""
        return [self.get_column_results(column) for column in self.dataframe.columns]

    def load_styles(self, stylefile_path):
        self.stylefile = StyleFile(stylefile_path)

    def get_column_results(self, column: str) -> ColumnResults:
        """
        :param column: the column to get the results for
        :return: a :class:`ColumnResults` object containing all the results for tests that tested the specified column
        """
        res_list = [result for result in self.results if column in result.columns_tested]
        return ColumnResults(column, res_list, self.dataframe, self.stylefile.get_column_style(column))

    def graph_validity_heatmap(self):
        """
        Generated a 1D heatmap of the columns by validity, color coded by the default integrity levels
        dictionary set for the database.

        In order for

        :return: The pyplot figure containing the graph
        """
        test_labels = [column for column in self.dataframe.columns if self.get_column_results(column).tested] \
                      + ['Dataframe']
        data = np.array(
            [result.num_valid / self.num_rows for result in self.column_results if result.tested]
            + [self.num_rows_valid / self.num_rows]
        )
        fig, ax = plt.subplots()
        step_colors, step_values = self.stylefile.dataframe_style.transposed
        color_map = utils.nonlinear_cmap(step_colors, step_values)

        seaborn.heatmap([data],
                        vmin=0, vmax=1,
                        cmap=color_map,
                        annot=True, fmt='.1%',
                        annot_kws={'rotation': 90},
                        xticklabels=test_labels, yticklabels=False)

        for t in ax.texts: t.set_text(t.get_text() + " %")
        return fig

    def graph_summary(self):
        """
        Adds and returns a pyplot figure containing two pie charts - one for tested vs. untested columns, and one for valid vs. invalid columns.
        """
        colors = [self.stylefile.dataframe_style.values[-1][0], self.stylefile.dataframe_style.values[0][0]]

        fig, (ax1, ax2) = plt.subplots(2)

        tested_data = [self.num_cols_tested, self.num_cols_untested]
        ax1.pie(tested_data, colors=colors, autopct=utils.make_autopct(tested_data))
        ax1.legend(['Tested', 'Untested'])

        valid_data = [self.num_rows_valid, self.num_rows_invalid]
        ax2.pie(valid_data, colors=colors, autopct=utils.make_autopct(valid_data), labels=['Valid', 'Invalid'])
        ax2.legend(["Valid", 'Invalic'])

        return fig

    def print(self, show_valid_cols=False, show_untested=False, stub=False, print_all_failed=False):
        """
        Produces a coverage report of the tests done — how many of the columns were tested, how many were valid,
        and a sample of invalid rows for each column.

        :param show_valid_cols: print result summary for columns that were completely valid. Default false.
        :param show_untested: show columns without tests. Default false.
        :param stub: don't print individual data for each column, just portions tested and valid. Default false.
        :param print_all_failed: print all rows where a test failed. By default only prints up to 10.
        """

        num_cols = len(self.dataframe.columns)

        print(f'Columns Tested: {self.num_cols_tested}/{num_cols} ({round(self.num_cols_tested / num_cols * 100)}%).')
        print(f'Columns valid: {self.num_cols_valid}/{self.num_cols_tested} ({round(self.num_cols_valid / self.num_cols_tested * 100, 2)}%).')

        if not stub:  # If stub not set, print details for individual columns.
            print()
            for i, column in enumerate(self.dataframe.columns, 1):
                column_res = self.get_column_results(column)
                # Don't show valid or untested columns unless specified.
                if (column in self.cols_checked or show_untested) \
                        and (column in self.cols_checked and not column_res.valid or show_valid_cols):
                    column_res.print(columns_to_include=[self.dataframe.columns[0]], column_number=i,
                                     print_all_failed=print_all_failed)
