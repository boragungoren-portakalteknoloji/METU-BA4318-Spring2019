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
def calculate(df):
    # Add columns for inventory level and value
    bought = numpy.asarray(df['Bought'])
    consumed = numpy.asarray(df['Consumed'])
    cost = numpy.asarray(df['Cost'])
    size = len (cost)
    invlevel = []
    invvalue = []
    # calculate inventory level at end of row
    lastlevel = 0 # initial inventory is always zero
    for index in range(0, size):
        # the rows are copies of original dataframe
        # First update inventory level
        currentlevel = lastlevel + bought[index] - consumed[index]
        invlevel.append( currentlevel)
        lastlevel = currentlevel # necessary for next iteration
        # Second calculate inventory value
        unitprice = cost[index] / bought [index]
        currentvalue = currentlevel * unitprice
        invvalue.append( currentvalue) 
    return invlevel,invvalue    

# Actual program
dataframe = process_file("Coffee Consumption.txt")
invlevel,invvalue = calculate(dataframe)
# Means of consumption, carried inventory, and inventory value
meanq = round(dataframe['Consumed'].mean(),2)
print("Mean level of consumption:", meanq)
meaninv = round(numpy.mean(invlevel),2)
print("Mean level of inventory carried:", meaninv)
meanval = round(numpy.mean(invvalue),2)
print("Mean level of inventory value:", meanval)
