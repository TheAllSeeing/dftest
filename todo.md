## Define the problem
**I want to say**
1. Cells - represent individual cells of a table -- maybe keep condition, maybe just column
2. Columns - data type? verifiers. Do they contain 
3. DataRow? basically `Dict[Column, Cell]`
4. Tables - I guess `List[DataRow]` or something like it. 
5. Database - set of tables I guess

_We want to start with Tables + Columns. More than that is not really necessary._

**BUT:**
Should I even create my own architecture for this? I guess I'll start with that.
I don't think this can really work beyond a very basic proof-of-context, since 
the whole point is to interject in between calls to "conventional" API. 
We are not planning to compete with pandas.


## Highly simplified scenario.
There is some table of data, and we want to preform basic tests on 
each column individually. This is actually easily possible with basic SQL,
but the more interesting thing is the coverage part -- I want to keep count 
of how many cells I've checked, and show some analysis. 
Again, if we super-simplify, just show a percentage of cells checked.

## The injection point
 - The test needs to run each during each change to the data. 
Though I don't think it was said explicitly, and I should mail Rani. 
We talked of two ways to do this:
   - Wrap an existing interface, e.g pandas. Implement all the same features but 
   rerun apropriate tests after modification
   - Use sys.settrace or pdb to run tests after calls to modification functions, or 
   something along these lines. My main confusion as to this is the manner with which 
   to detect these functions, unless we just support specific existing libraries or 
   create them completely on our own (which we won't because I really do not think
   we plan to compete with pandas)


## Now,
having generally done a simplified version, or at least figured out 
how to easily do it.