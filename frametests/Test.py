# Sets type hints to act as string so I can reference a class (in a type hint) before it's defined.
from __future__ import annotations

# For autodetecting accessed columns as a test run, and set the trace function back afterwards.
from sys import settrace, gettrace
# For better typehinting
from typing import List, Callable, Tuple, Set, Hashable, Union, Any
# For better type hinting, and detecting uses of Series.__getitem__ specifically.
from pandas import DataFrame, Series, Index
# For autodecteting test result type
from numpy import bool_
# for chekcing if result is number of invalid line
from numbers import Number


class Test:
    """
    A named predicate to enact on the rows of a :class:`pandas.DataFrame` which test the
    validity of one or more of its columns.
    """

    def __init__(self, predicate: Callable[[DataFrame], List[Hashable]], column_index: Index = None, name: str = None,
                 tested_columns: List[str] = None, ignore_columns: List[str] = None, success_threshold=1):
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
        self.success_threshold = success_threshold

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
        result, columns_tested = self.test(dataframe)
        if isinstance(result, bool) or isinstance(result, bool_):
            return BooleanTestResult(self, columns_tested, len(dataframe.index), result)
        elif isinstance(result, Number):
            return NumberTestResult(self, columns_tested, len(dataframe.index), result)
        elif isinstance(result, list):
            return IndexTestResult(self, columns_tested, len(dataframe.index), result, self.success_threshold)
        else:
            raise ValueError(f'Test {self.name}: Invalid test return type: {type(result).__name__} {str(result)}')


class TestResult:
    """
    The results of a :class:`Test` preformed on a dataframe.
    """

    def __init__(self, origin_test: Test, columns_tested: Set[str], num_tested: int,
                 result: Union[bool, Number, List[Hashable]]):
        """
        :param origin_test: the test this is a result of
        :param columns_tested: the columns that were tested on the dataframe (by name)
        :param result: a test result - either a list of invalid row index, or a number of invalid rows
        """
        self.from_test = origin_test
        self.columns_tested = columns_tested
        self.num_tested = num_tested
        self.result = result


class BooleanTestResult(TestResult):
    def __init__(self, origin_test: Test, columns_tested: Set[str], num_tested: int, result: bool):
        super(BooleanTestResult, self).__init__(origin_test, columns_tested, num_tested, result)
        self.success = result


class NumberTestResult(TestResult):
    def __init__(self, origin_test: Test, columns_tested: Set[str], num_tested: int, result: Number):
        super(NumberTestResult, self).__init__(origin_test, columns_tested, num_tested, result)
        self.num_invalid = self.num_tested - result
        self.success = result

    @property
    def num_valid(self):
        """
        Number of dataframe rows that passes the test
        """
        return self.num_tested - self.num_invalid


class IndexTestResult(TestResult):
    def __init__(self, origin_test: Test, columns_tested: Set[str], num_tested: int, result: List[Hashable], success_threshold: float):
        super(IndexTestResult, self).__init__(origin_test, columns_tested, num_tested, result)
        self.num_invalid = len(result)
        self.invalid_row_index = result

        self.success = self.num_valid / self.num_tested >= success_threshold

    @property
    def num_valid(self):
        """
        Number of dataframe rows that passes the test
        """
        return self.num_tested - self.num_invalid

    def get_invalid_rows(self, src_dataframe: DataFrame) -> DataFrame:
        """
        :param src_dataframe: the dataframe the test was ran on
        :return: a dataframe containing each of the rows invalid under this test
        """
        return src_dataframe.iloc[self.invalid_row_index]