def calculateAverage(listOfDatapoints):
    sum = 0
    for datapoint in listOfDatapoints: # repeats tasks for all items in list
        sum = sum + datapoint
        avg = sum / len(listOfDatapoints)
    return avg

phlevels = [7.1, 7.5, 7.3, 6.9, 7.2, 7.4, 7.2, 7.4, 6.9, 6.8, 5.0, 5.1, 5.9]
length = len(phlevels)
avgOld = calculateAverage(phlevels[:length-3])
avgLatest = calculateAverage(phlevels[length-3:])
devAbs = abs(avgLatest - avgOld)
devPercent = devAbs / avgOld
if devPercent > 0.10: # conditional code execution
    print("Quality control alarm!")
else:
    print("All is OK")
