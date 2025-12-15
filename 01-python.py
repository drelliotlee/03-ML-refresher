# Here I am going through `https://learnxinyminutes.com/python/` and 
# just listing everything I forgot in this notebook

# STRINGS 
s = "string"
len(s)
f"this is a f-{s}"


# LISTS
mylist=['a','b','c','d']
mylist[0]           # a (index -> element)
mylist.index('c')   # 2 (element -> index)
'e' in mylist       # False (check membership)


# TUPLES
x,y,z = (1,2,3) # unpacking


# DICTIONARIES
d = {'k1':'a', 'k2':'b'}
d['k1']                      # 'value1' (lookup)
'k3' in d                    # False    (check membership)
list(d.keys())
list(d.values())


# IF SYNTAX
if x > y:
    pass
elif y > z:
    pass
else:
    pass


# FOR SYNTAX
for x in mylist:
    pass
for i,x in enumerate(mylist):
    pass


# TRY SYNTAX
try:
    pass
except Exception as e:
    print("Error:", e)


# FUNCTIONS
def my_function(x, y='default_value'):
    return x + y
def my_function2(*args, **kwargs):
    # args is a tuple of unnamed inputs
    # kwags is a dict of named inputs
    return args, kwargs
(lambda x: x**2)(5)  # like f(5)=25

# LIST COMPREHENSIONS
my_list = [f(x) for x in someList if condition(x)]

# IMPORTANT
#importing a file in the working directory just runs the file
import pandas          # usage: pandas.DataFrame()
import pandas as pd    # usage: pd.DataFrame()
from pandas import *   # usage: DataFrame()

if __name__ == "__main__":
    # code here runs only if this file is executed directly,
    # not if it is imported as a module
    pass

