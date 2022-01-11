from pandas import Series


def boolean_test(column: str, row: Series):
    return row[column] == 'Yes' or row[column] == 'No'


def nonzero_natural_test(column, row):
    return str(row[column]).isdigit() and int(row[column]) > 0


def percent_test(column, row):
    try:
        return 0 <= int(row[column]) <= 100
    except ValueError:
        return False


def reasonable_room_cost_range_test(row):
    return 100 < row['Room.Board'] < 10000


def apps_accept_enroll_test(row):
    return row['Apps'] >= row['Accept'] >= row['Enroll']


def enroll_range_test(row):
    return 0 < row['Enroll'] <= 7500


def sane_spending_test(row):
    try:
        return int(row['Personal']) < 4000
    except ValueError:
        return False
