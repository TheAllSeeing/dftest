## Features
- Allow adding more general tests for a specific datatypes

## Chores
- Use `dataframe.itertuples()` instead of `dataframe.iterrows()` when testing 
  over a dataframe. `itertuples` is more efficient and preserves dtypes 
  (though it returns a namedtuple instead of a Series).
- Don't iterate over all the rows for each test, iterate once and run all 
  tests over it. Currently, the Series (or tuple) objects for each row are 
  created again and again and again.