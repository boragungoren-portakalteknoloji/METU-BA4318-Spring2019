# Solution for group at 09:00
# Find a small constant order level where we never run out of coffee
# Also calculate the total cost of orders and compare to actual total cost
import pandas
import numpy

# Function that reads from file and returns column names and dataframe
def process_file(filename):
    frame = pandas.read_csv(filename, delim_whitespace=True)
    return frame

# Function that checks inventory levels for each month
# given the constant order amount
def try_quantity(df, quantity):
    # Add columns for inventory level and value
    level = 0 # 
    for index, row in df.iterrows():
        # ordering quantity at all months
        level = level + quantity - row['Consumed'] 
        if level < 0:
            return False # Ooops! We ran out of coffee, 
    # If we are here that means we did not return false in any row
    # That means we did not run out of coffee
    # print("q =",quantity,"is OK")
    return True

def search_quantity(df):
    q = 0
    found = False
    while found == False:
        # try quantity
        q = q + 1 
        found = try_quantity(df,q)
    return q

def calculate_result(df,q):
    totalcost = 0
    for index, row in df.iterrows():
        # Each month has a different unit price
        unitprice = row['Cost'] / row['Bought']
        totalcost = totalcost + q * unitprice
    return totalcost  

# Actual program
df = process_file("Coffee Consumption.txt")
# print(df)
# Calculate current order level and cost
sumorders = df['Bought'].sum()
sumcost = round ( df['Cost'].sum(), 2)
# Search for the minimum level where we do not run out of coffee
minq = search_quantity(df)
# Calculate total orders and cost
sumq = len(df) * minq
costq = round( calculate_result(df,minq), 2)
# Print results    
print("Actual ordering method ends up ordering",
        sumorders, "grams coffee at total cost of", sumcost)
print("A constant order level of", minq, " results in ordering",
      sumq, "grams coffee at total cost of", costq)