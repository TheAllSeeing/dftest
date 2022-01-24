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

    columns_to_test = set(df.columns) - {'Metadata Date'}

    dbtests.add_generic_test(tests.is_not_null, columns=columns_to_test)

    # Note that because of the dataset significant inconsistency, we have
    # to make sure to catch ValueErrors when converting to int or float for
    # calculations.
    def try_parse_int(int_str):
        try:
            return int(int_str)
        except ValueError:
            return -1


    def years_match(dataframe):
        bool_arr = dataframe['Object Number'].str.extract('([0-9]{2})\.*')[0].apply(try_parse_int) != (
                dataframe['AccessionYear'].apply(try_parse_int) % 100)
        return [i for i, check in enumerate(bool_arr) if check]


    dbtests.add_test(years_match)

    results = dbtests.run()
    print([result.from_test.name for result in results.results])

    # bool_cols = ['Private']
    # tests.add_generic_test(test_funcs.boolean_test, name='Yes/No Compliance', columns=bool_cols)
    # results.
    # results.get_column_results('Personal').tests_heatmap()
    results.graph_summary()
    results.graph_validity_heatmap()
    col_results = results.get_column_results('Object Number')
    col_results.graph_validity_heatmap()
    results.plt.show()

    # print(MixedDataTypes().add_condition_rare_type_ratio_not_in_range().run(df))
