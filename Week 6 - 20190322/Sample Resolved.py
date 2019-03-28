import random
import pandas
import numpy

# Read data from file into pandas dataframe
def get_df(filename, colname):
    df = pandas.read_csv(filename)
    minvalue = df[colname].min()
    maxvalue = df[colname].max()
    size = len(df)
    return minvalue, maxvalue, size, df

colname = 'Data'
samplemin, samplemax, samplesize, dataframe = get_df("C:\\Users\\gbora\\Desktop\\data.csv", colname) 

# Create list of random values, size = dataframe
# would also need min and max values 

def create_random_values(minv=0.0, maxv=100.0, size=100):
    randoms = []
    for count in range(0,size):
        randoms.append( random.uniform(minv, maxv) )
    return randoms
randomvalues = create_random_values(samplemin, samplemax, samplesize)

# export from dataframe into a list
actualvalues = dataframe[colname].tolist()

# calculate coefficient
# list sizes should be the same
coefficient = numpy.corrcoef(actualvalues, randomvalues)[0,1]
print("Coefficient of correlation with random data:", coefficient)