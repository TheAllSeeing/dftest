# For getting a dataframe in testing (via read_csv) and setting dataframe to not print dimensions (via options
# attribute). from-import not used to make the purpose of these methods more explicit.
import pandas

# Example testing functions defined by me
import test_funcs
from DBTests import DBTests

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

    results = tests.run()
    results.print()
