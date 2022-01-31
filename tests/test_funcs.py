import dftests

yesno_test = dftests.in_list_test(['yes', 'no'])
percent = dftests.in_range_test(0, 100, True, True)


# Note that because of the dataset significant inconsistency, we have
# to make sure to catch ValueErrors when converting to int or float for
# calculations.
def try_parse_int(int_str):
    try:
        return int(int_str)
    except ValueError:
        return -1


def years_match(dataframe):
    bool_arr = dataframe['Object Number'].str.extract('(?:[0-9]{2})*([0-9]{2})')[0].apply(try_parse_int) != (
            dataframe['AccessionYear'].apply(try_parse_int) % 100)
    return [i for i, check in enumerate(bool_arr) if check]