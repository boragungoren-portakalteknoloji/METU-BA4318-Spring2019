import pandas as pd
import numpy as np
df = pd.read_csv("dataset.txt") # reads into a dataframe
print(df)
# print(df.axes)
# print(df.index)
# print(df.columns)

# access individual column
print ("The name column.")
# print(df["Name"])
print("Has size:", df.Name.size, " , has objects of type", df.Name.dtype )
print("Unique names:", df.Name.unique())

print("The coffee consumption")
print("Average:", df.Coffee.mean())
print("The first two lines have data:")
print(df.Coffee.head(2) )
print("The last two lines have data:")
print(df.Coffee.tail(2) )

coffeearray = np.asarray(df.Coffee)
print(coffeearray)
# [7 3 4 4 3 0]

print( df.groupby("Name")["Coffee"].mean() )
#             ^ Filter


