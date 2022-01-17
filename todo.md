## Features
- Display graphs
- Create some common tests to be available immediately
- Add default tests
- Read config file to define tests, instead of manually defining them
- Add color-coded levels of integrity 

## Chores
- Use `dataframe.itertuples()` instead of `dataframe.iterrows()` when testing 
  over a dataframe. `itertuples` is more efficient and preserves dtypes 
  (though it returns a namedtuple instead of a Series).
