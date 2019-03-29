import random
import numpy
import pandas

def get_random_inrange(min=0.0, max=100.0, size=100):
    randoms = []
    for count in range(0,size):
        rand = random.uniform(min,max)
        randoms.append(rand)
    return randoms

list1 = [] #list of numbers to be extracted from dataframe
list2 = [] #list of random numbers to be created

dataframe = pandas.read_csv("C:\\Users\\gbora\\Desktop\\data.csv")
print(dataframe.axes)
list1 = dataframe['Data'].tolist()

min = dataframe['Data'].min()
max = dataframe['Data'].max()
size = len(list1)
list2 = get_random_inrange(min, max, size)

coefficient = numpy.corrcoef(list1, list2)[0,1]
print("Coefficient of correlation with random data:", coefficient)
significant = 0.4
if abs(coefficient) < significant:
    print("No significant")
else:
    print("Significant")

