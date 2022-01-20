import pandas
from matplotlib import pyplot

import test_funcs
import tests
from DBTests import DBTests

if __name__ == '__main__':
    # Download from https://github.com/metmuseum/openaccess/raw/master/MetObjects.csv
    df = pandas.read_csv('MetObjects.csv')
    dbtests = DBTests(df)

    # with open('config.json') as conf_file:
    #     tests.load_config(conf_file)

    dbtests.add_generic_db_test(tests.is_not_null, column_autodetect=True)


    # bool_cols = ['Private']
    # tests.add_generic_test(test_funcs.boolean_test, name='Yes/No Compliance', columns=bool_cols)
    #
    # natural_cols = set(df.columns) - {'Private', 'S.F.Ratio', 'Unnamed: 0'}
    # tests.add_generic_test(test_funcs.nonzero_natural_test, name='Integer Compliance', columns=natural_cols)
    #
    # perc_cols = ['Top10perc', 'Top25perc', 'PhD', 'Terminal', 'perc.alumni', 'Grad.Rate']
    # tests.add_generic_test(test_funcs.percent_test, name='Percent Compliance', columns=perc_cols)
    #
    # tests.add_test(test_funcs.reasonable_room_cost_range_test)
    # tests.add_test(test_funcs.apps_accept_enroll_test, name='Apps > Accept > Enroll')
    # dbtests.add_test(test_funcs.sane_spending_test, name='Sane Spending')

    results = dbtests.run()
    # results.print()
    # results.show_summary()
    # res = results.get_column_results('Personal')
    # res.print()
    # results.
    # results.get_column_results('Personal').tests_heatmap()
    results.graph_validity_heatmap()
    results.plt.show()

    # print(MixedDataTypes().add_condition_rare_type_ratio_not_in_range().run(df))
