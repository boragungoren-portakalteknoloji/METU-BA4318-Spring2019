# Solution for group at 09:00
# Calculate and print mean value of inventory carried on to next month
# And its value
import pandas
import numpy

# Function that reads from file and returns column names and dataframe
def process_file(filename):
    frame = pandas.read_csv(filename, delim_whitespace=True)
    return frame

# Function that adds a named column to a dataframe
def add_column(df, colname, defaultval=numpy.nan):
    # create new column and initialize with defaultval
    df[colname] = defaultval
    return df

# function to calculate carried over inventory
def add_carried(df, invcolumn='Inventory Level', valcolumn='Inventory Value'):
    # Add columns for inventory level and value
    df = add_column(df, invcolumn)
    df = add_column(df, valcolumn)
    # calculate inventory level at end of row
    lastlevel = 0 # initial inventory is always zero
    for index, row in df.iterrows():
        # the rows are copies of original dataframe
        # First update inventory level
        currentlevel = lastlevel + row['Bought'] - row['Consumed']
        df.loc[index, invcolumn] = currentlevel
        lastlevel = currentlevel # necessary for next iteration
        # Second calculate inventory value
        unitprice = row['Cost'] / row['Bought']
        currentvalue = currentlevel * unitprice
        df.loc[index,valcolumn] = currentvalue
    return df    

# Actual program
dataframe = process_file("Coffee Consumption.txt")
dataframe = add_carried(dataframe)
# Means of consumption, carried inventory, and inventory value
meanq = round(dataframe['Consumed'].mean(),2)
print("Mean level of consumption:", meanq)
meaninv = round(dataframe['Inventory Level'].mean(),2)
print("Mean level of inventory carried:", meaninv)
meanval = round(dataframe['Inventory Value'].mean(),2)
print("Mean level of inventory value:", meanval)
