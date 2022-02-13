import pandas

from dftest import tests
from dftest.DBTests import DBTests

if __name__ == '__main__':
    # Download from https://github.com/metmuseum/openaccess/raw/master/MetObjects.csv
    df = pandas.read_csv('MetObjects.csv')
    dbtests = DBTests(df)

    object_num_test = dftests.match_test(r'([0-9]{2}|[0-9]{4}).[0-9]{1,4}.[0-9]{1,4}')

    columns_to_test = set(df.columns) - {'Metadata Date'}
    dbtests.load_config('tests.conf')

    results = dbtests.run()
    results = dbtests.run()

    # results.print()

    results.graph_summary()
    results.graph_validity_heatmap(binary=True)
    col_results = results.get_column_results('AccessionYear')
    col_results.open_invalid_rows()
    col_results.graph_validity_heatmap()
    results.plt.show()