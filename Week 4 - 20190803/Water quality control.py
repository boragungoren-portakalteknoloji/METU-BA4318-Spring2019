def calculateAverage(listOfDatapoints):
    sum = 0
    for datapoint in listOfDatapoints: # repeats tasks for all items in list
    sum = sum + datapoint
    avg = sum / len(listOfDatapoints)
    return avg

phlevels = [7.1, 7.5, 7.3, 6.9, 7.2, 7.4, 7.2, 7.4, 6.9, 6.8, 5.0, 5.1, 5.9]
# calculate avg of all data except past (latest) three
length = len(phlevels)
# part with indexes 0, 1, ..., length-4 (inclusive)
olddata = phlevels[0:length-3] # start from zero less than length-3
# olddata = phlevels[:length-3] # first one empty means from start
print(olddata)
avgOld = calculateAverage(olddata)
# calculate avg of past (latest) three
latestdata = phlevels[length-3:] #second one empty means to the end
print(latestdata)
avgLatest = calculateAverage(latestdata)
# calculate deviation (absolute)
devAbs = abs(avgLatest - avgOld)
# calculate deviation (percent)
devPercent = devAbs / avgOld
# if deviation is > %10 (0.10) then sound an alarm
if devPercent > 0.10: # conditional code execution
    print("Quality control alarm!")
else:
    print("All is OK")
