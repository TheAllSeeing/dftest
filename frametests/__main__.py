#!/bin/python3
import argparse
import os.path
from configparser import ConfigParser

import pandas

from frametests import utils
from frametests.DBTests import DBTests

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(prog='dftest', formatter_class=argparse.RawTextHelpFormatter)
    arg_parser.add_argument('-c', '--config', metavar='CONFIG_FILE', required=False, nargs=1, help='specify configuration file')
    arg_parser.add_argument('--dataframe', metavar='DATAFRAME', required=True, help='Specify either \n'
                                                                     '- python path to DataFrame object\n'
                                                                     '- python path to DataFrame supplier function\n'
                                                                     '- file path a file with a dataframe in any legal pandas format\n')
    arg_parser.add_argument('files', metavar='FILE_OR_DIR', nargs='+', help='files or dirs to search for tests')

    args = arg_parser.parse_args()

    if args.config is not None:
        config = ConfigParser()
        config.read(args.config)
    else:
        config = None

    if os.path.isfile(args.dataframe):
        if args.dataframe.endswith('.csv'):
            df = pandas.read_csv(args.dataframe)
        elif args.dataframe.endswith('.tsv'):
            df = pandas.read_csv(args.dataframe, delimiter='\t')
        elif args.dataframe.endswith('.json'):
            df = pandas.read_json(args.dataframe)
        elif args.dataframe.endswith('.xlsx'):
            df = pandas.read_excel(args.dataframe)
        else:
            raise ValueError('Invalid dataframe file: unrecognized extension: ' + args.dataframe)

    else:
        df_obj = utils.get_func_from_addr(args.dataframe)
        if callable(df_obj):
            df = df_obj()
        else:
            df = df_obj

    file_args = args.files
    files = []
    for filename in file_args:
        if os.path.isdir(filename):
            for dirpaths, dirs, files in os.walk('.', topdown=True):
                files += [os.path.join(dirpath, file) for dirpath, file in zip(dirpaths, files) if file.endswith('.py')]
        elif os.path.isfile(filename) or os.path.islink(filename):
            files.append(filename)

    test_funcs = []
    for filename in files:
        module = __import__(filename.replace(os.path.pathsep, '.'))
        test_funcs_to_add = [getattr(module, attr) for attr in dir(module) if attr.startswith('test')]
        test_funcs_to_add = list(filter(lambda attr: callable(attr), test_funcs_to_add))  # make sure to only take funcs
        test_funcs += test_funcs_to_add

    dftests = DBTests(df)

    for func in test_funcs:
        if func.__code__.co_argcount == 1:
            if func.__name__ in config.sections():
                args = config[func.__name__]
                name = args.pop('name', None)
                tested_columns = utils.read_config_list(args.pop('tested_columns', None))
                ignore_columns = utils.read_config_list(args.pop('ignore_columns', None))
                success_threshold = args.pop('success_threshold', None)
                dftests.add_test(func, name, tested_columns, ignore_columns, success_threshold, **args)
            else:
                dftests.add_test(func)
        elif func.__code__.co_argcount == 2:
            if func.__name__ in config.sections():
                args = config[func.__name__]

                include = utils.read_config_list(args.pop('include', None))
                include_dtypes = utils.read_config_list(args.pop('include_dtypes', None))
                exclude = utils.read_config_list(args.pop('exclude', None))
                name = args.pop('name', None)

                column_autodetect = args.getboolean('column_autodetect', False)
                if column_autodetect in args.keys():
                    del args['column_autodetect']

                ignore_columns = utils.read_config_list(args.pop('ignore_columns', None))

                success_threshold = args.pop('success_threshold', None)
                success_threshold = success_threshold if success_threshold is None else float(success_threshold)

                dftests.add_generic_test(func, include, include_dtypes, exclude, name, column_autodetect,
                                         ignore_columns, success_threshold, **args)
            else:
                dftests.add_generic_test(func)
