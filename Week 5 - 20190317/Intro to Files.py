# The file name is for a file in the same folder
filename = "dataset.txt"
infile = open(filename, "r") # r for reading, w for writing, a for appending
if infile.readable == True:
    namesline = infile.readline()
    nameslist = namesline.split(";")
    # print("column names:", namesline)
    print("names list:", nameslist)
    for index in range(0,5):
        currentline = infile.readline()
        valueslist = currentline.split(";")
        #print("current line: ", currentline)
        print("values list:", valueslist)
    infile.close() # do not forget
    

