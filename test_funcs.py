from pandas import Series


def boolean_test(column: str, row: Series):
    return row[column] == 'Yes' or row[column] == 'No'


def integer_test(column, row):
    return str(row[column]).isdigit()


def tenpercent_test(row):
    return 0 < row['Top10perc'] < 50
