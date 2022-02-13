import numpy as np
import pandas as pd

from dftest.DBTests import DBTests

if __name__ == '__main__':
    df = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list('ABCD'))
    df['E'] = np.random.choice([chr(i) for i in range(97, 121)], size=100)

    dbtests = DBTests(df)

    def percent_test(column: str, dataframe: pd.DataFrame):
        valid_arr = dataframe[column].apply(lambda x: 0 <= x <= 100)
        return [i for i, cell in enumerate(valid_arr) if not cell]


    dbtests.add_generic_test(percent_test, include=df.select_dtypes(int).columns)

    results = dbtests.run()
    results.print()
    results.graph_validity_heatmap()
    results.plt.show()
