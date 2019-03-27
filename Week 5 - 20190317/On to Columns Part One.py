# Two lists, zipped together can function as two columns
nameslist = ["Bora", "Tufan", "Ali Cem", "Oğuzhan", "İlker"]
consumptionlist = [7, 3, 4, 4, 0]

zipped = zip(nameslist,consumptionlist)
for name,coffee in zipped:
    print(name, " has consumed ", coffee, "cups today.")
    
# Dictionaries can be used to create an indexed row structure
# If each value in a dictionary is a list,
# then we can have row-indexed structure similar to a matrix
# The items in the list (row) are indexed, so we can access
# individual items
dictionary = { "names" : nameslist, "coffee" : consumptionlist}
print("List of names mapped:", dictionary["names"])
print("First item:", dictionary["names"][0])

for key in dictionary:
    print("Current key is:", key)
    for item in dictionary[key]:
        print("An item in list mapped to key", key, "is ", item)
    
for key in dictionary:
    print("Current key is:", key)
    size = len(dictionary[key])
    for index in range(0,size): # range starting from 0 less than size
        item = dictionary[key][index]
        print("An item in list mapped to key", key, "is ", item)

# If we look at the same from the other direction
# We could decide to see the indexed direction as columns
# Then each value (list) in the dictionary is a column
# Hence the index of said list is the row-index 

# Create 4 x 4 matrix i=0...3 to j=0..3  s.t. (i,j)th elementh value is i+j
matrix = {}
for i in range(0,3):
    # create the row as a list
    currentrow = []
    #populate row
    for j in range(0,3):
        currentrow.append(i+j)
    # add row to matrix
    matrix[i] = currentrow
print(matrix)
# {0: [0, 1, 2], 1: [1, 2, 3], 2: [2, 3, 4]}

#swap rows
row0 = matrix[0]
row1 = matrix[1]
matrix[0] = row1
matrix[1] = row0
print(matrix)
# {0: [1, 2, 3], 1: [0, 1, 2], 2: [2, 3, 4]}
    
