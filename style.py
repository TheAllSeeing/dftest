from __future__ import annotations

import builtins
import json
import operator
from functools import partial, reduce
from typing import Dict, List, Tuple

from pandas import DataFrame, Index

import tests
import utils
from Test import Test


class Style:
    def __init__(self, raw: List[Dict[str, float]] = None):
        if raw is None:
            self.values = [('red', 0),
                           ('orange', 0.25),
                           ('yellow', 0.5),
                           ('blue', 0.75),
                           ('green', 1)]
        else:
            self.values = [(list(item.keys())[0], list(item.values())[0]) for item in raw]

        self.transposed = tuple(map(list, zip(*self.values)))


    def colorcode(self, validity_rate: float):
        color_code = None
        for color, step in self.values:
            if validity_rate >= step:
                color_code = color
            else:
                return color_code
        return 'grey' if color_code is None else color_code  # no valid integrity-levels specified or passed all of them


class StyleFile:
    def __init__(self, filename=None):
        if filename is not None:
            with open(filename, 'r') as stylefile:
                parsed = json.load(stylefile)
        else:
            parsed = {}

        if '__DATAFRAME__' in parsed.keys():
            self.dataframe_style = Style(parsed['__DATAFRAME__'])
            del parsed['__DATAFRAME__']
        else:
            self.dataframe_style = Style()

        if '__DEFAULT__' in parsed.keys():
            self.default_style = Style(parsed['__DEFAULT__'])
            del parsed['__DEFAULT__']
        else:
            self.default_style = Style()

        self.styles = {column: Style(palette) for column, palette in parsed.items()}

    def get_column_style(self, column):
        return self.styles.get(column, self.default_style)
