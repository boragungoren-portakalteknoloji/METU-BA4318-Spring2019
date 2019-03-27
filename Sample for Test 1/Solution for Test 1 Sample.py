import pandas
import numpy

# Function that reads from file and returns values in array
def process_file(filename):
    df = pandas.read_csv(filename, delim_whitespace=True)
    temparray = numpy.asarray(df['Temperature'])
    return temparray

# Function to calculate points
def calculate_points(temp):
    hotpoints = 0
    coldpoints = 0
    if temp < 8.0 :
        # set coldpoints for a cold day
        coldpoints = 8.0 - temp
    elif temp > 16.0 :
        # set hotpoints for a hot day
        hotpoints = temp - 16.0
    # If the temp was between 8 and 16 then both points will remain 0
    # print statement to debug 
    # print("temp:", temp, "cold, hot:", coldpoints,hotpoints)
    return coldpoints, hotpoints
    
# function to calculate averages
def calc_avgpoints(tarray, nyears):
    totalcold = 0
    totalhot = 0
    # for loop for all days in 1951 to 2018
    for temp in tarray:
        # get points for current day
        cold, hot = calculate_points(temp)
        totalcold = totalcold + cold
        totalhot = totalhot + hot
    # nyears years
    avgcold = totalcold / nyears
    avghot = totalhot / nyears
    return avgcold, avghot

# Actual program calling functions
tarray = process_file("Schiphol.txt") # Array for 1958 to 2018
avgcold, avghot = calc_avgpoints(tarray,nyears=68) # total of 68 years
print("Average for 68 years. Coldpoints:", avgcold, " Hotpoints: ", avghot)