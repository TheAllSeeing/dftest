# Sets type hints to act as string so I can reference a class (in a type hint) before it's defined.
from __future__ import annotations

# For autodetecting accessed columns as a test run, and set the trace function back afterwards.
from ast import Index
from sys import settrace, gettrace
# For better typehinting
from typing import List, Callable, Tuple, Set, Hashable, Union, Any
# For better type hinting, and detecting uses of Series.__getitem__ specifically.
from pandas import DataFrame, Series


class Test:
    """
    A named predicate to enact on the rows of a :class:`pandas.DataFrame` which test the
    validity of one or more of its columns.
    """

    def __init__(self, predicate: Callable[[DataFrame], List[Hashable]], column_index: Index = None, name: str = None,
                 tested_columns: List[str] = None, ignore_columns: List[str] = None):
        """
        :class:`Test` class constructor.

        By default, the class will autodetect which columns are being tested as it runs over
        the database. This is fairly crude though, and detects any columns which the test accesses.
        The `tested_columns` and `ignore_columns` parameters can be used to override or modify this behaviour.

        :param predicate: the predicate which will be used to test the rows of the dataframe in this test.

        :param name: a name for the test which will be displayed when running it and can be accessed via the `name`
        property. By default (and if given `None`) this will be set to the name of the predicate function.

        :param tested_columns: setting this parameter to a list of columns will override this behaviour and cause the
        run method to return the given list as the columns tested.

        :param ignore_columns: columns to ignore in tested-columns autodetection. If `tested_columns` is set,
        columns that appear in it and in here will be ignored also.
        """
        self.predicate = predicate
        self.column_index = column_index

        if name is None:
            self.name = self.predicate.__name__
        else:
            self.name = name

        self.tested_columns = set() if tested_columns is None else set(tested_columns)
        self.ignore_columns = set() if ignore_columns is None else set(ignore_columns)

        if column_index is None and (tested_columns is None or len(tested_columns) == 0):
            raise ValueError("Tried to initialize test with no column index and no specified test columns!")

    def test(self, test_target: Any) -> Tuple[Union[bool, List[Hashable]], Set[str]]:
        """
        :param test_target: the row to test
        :returns: a tuple of the boolean test result and the set of columns that were tested.
        """

        # If tested_columns is set then no autodetection of tested columns is needed, just test the row
        # and return the defined tested columns (minus ones specified to ignore).
        if len(self.tested_columns) > 0:
            return self.predicate(test_target), self.tested_columns.difference(self.ignore_columns)

        # Else, the autodetecting tested columns is needed.
        # This is done via initializing an empty accessed_columns
        # set, then running the test with a tracing function which
        # detects accessed columns and adds them to the set

        accessed_columns = set()

        # More info about sys.settrace and the parameters of trace functions can be found
        # at https://explog.in/notes/settrace.html
        def add_accessed_columns(frame, event, arg):
            # Run on *calls* of *__getitem__* with a *Series* object *which is the test row*
            if event == 'call':
                if frame.f_code == Index.__getitem__.__code__:
                    accessed_columns.add(self.column_index[frame.f_locals['key']])
                if frame.f_code == Series.__getitem__.__code__:
                    key = frame.f_locals['key']
                    if key in self.column_index:
                        accessed_columns.add(key)

        original_trace = gettrace()
        settrace(add_accessed_columns)

        # Make sure to eventually set the trace function back.
        try:
            test_result = self.predicate(test_target)
        finally:
            settrace(original_trace)

        return test_result, accessed_columns.difference(self.ignore_columns)

    def run(self, dataframe: DataFrame) -> TestResult:
        invalid_index, columns_tested = self.test(dataframe)
        return TestResult(self, columns_tested, len(dataframe.index), invalid_index)


class TestResult:
    """
    The results of a :class:`Test` preformed on a dataframe.
    """

    def __init__(self, origin_test: Test, columns_tested: Set[str], num_tested: int,
                 invalid_rows_index: List[Hashable]):
        """
        :param origin_test: the test this is a result of
        :param columns_tested: the columns that were tested on the dataframe (by name)
        :param invalid_rows_index: a list of indexes for the rows where the test failed.
        """
        self.from_test = origin_test
        self.columns_tested = columns_tested
        self.num_tested = num_tested
        self.invalid_row_index = invalid_rows_index

    @property
    def success(self):
        """Whether the dataframe passed the test completely"""
        return len(self.invalid_row_index) == 0

    @property
    def num_invalid(self):
        """
        Number of dataframe rows that failed the test
        """
        return len(self.invalid_row_index)

    @property
    def num_valid(self):
        """
        Number of dataframe rows that passes the test
        """
        return self.num_tested - self.num_invalid

    def get_invalid_rows(self, src_dataframe: DataFrame) -> DataFrame:
        return src_dataframe.iloc[self.invalid_row_index]
