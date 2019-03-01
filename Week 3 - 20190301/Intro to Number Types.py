# Let's work with real numbers i.e. 1.5 2.7 etc
i = 1
j = 2
k = 3
x = k / j
print("x is ", x)
i = x / 3
print("i is ", i)
x = int( k / j ) # Because I want the whole part, I explicitly convet to int
print("x converted to integer is ", x)
x = int (1.6) # Supposed to be rounded to 2, has whole part 1
print("x from 1.6 is", x)
x = int (-1.6) # Supposed to be rounded to -2, has whole part -1
print("x from -1.6 is", x)
x = float(100)
print("x converted to float is", x)
zero = 0
# x = i / zero # divide by zero crashes program
