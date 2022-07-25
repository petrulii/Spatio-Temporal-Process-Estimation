import pyreadr

result = pyreadr.read_r('/path/to/file.RData') # also works for Rds

# done! let's see what we got
# result is a dictionary where keys are the name of objects and the values python
# objects
print(result.keys()) # let's check what objects we got
df1 = result["df1"] # extract the pandas data frame for object df1
df1.head()