from sys import argv, exit
from statistics import mean, stdev, median
from scipy.stats import sem
from pickle import load
import matplotlib.pyplot as plt

#   COMMAND-LINE ARGUMENTS: path plotType traceLength browser [# of traces to be plotted] 'one' or 'multi' 

def main():

    if len(argv)!= 6:
        exit("Takes 5 command-line arguments: path plot tracelength browser one/multi")
    else:
        filePath = argv[1]
        plotType = argv[2]
        traceLength = argv[3]
        browser = argv[4]
        quantity = argv[5]
    
    # Verify path exists
    try:
        with open(filePath, "rb") as f:
            pass
    except FileNotFoundError:
        exit(f"{filePath} does not exist.")

    # Unpickle trace file, quantity = number of traces to be returned in traceSet (the output of the unpickle function)
    unpklTrace = unpickle(filePath, quantity)

    # (Typical) parameters for plots, MAY BE CHANGED depending on plot; same with labels
    # default infoTag looks like "Chrome - https://site.com - 10 seconds - 50 run(s)"
    timeUnits = "ms"
    yAxisUnits = "Counter values"
    timeAxis = range(int(traceLength) * 1000)
    infoTag = f"{browser.capitalize()} - {unpklTrace[1]} - {traceLength} seconds - {unpklTrace[2]} run(s)"
    binNumber = 30
    average = False
    default = True

    if plotType == "scatter":
        for t in unpklTrace[0]:
            plt.scatter(timeAxis, t, s=2)

    if plotType == "average_scatter":
        plt.scatter(timeAxis, averager(unpklTrace[0]), color="orange", s=3)
        average = True

    if plotType == "line":
        for t in unpklTrace[0]:
            plt.plot(timeAxis, t, lw=1)
    
    if plotType == "average_line":
        plt.plot(timeAxis, averager(unpklTrace[0]), color="orange", lw=1)
        average = True

    if plotType == "histogram":
        for t in unpklTrace[0]:
            plt.hist(t, bins=binNumber)

        # Specific histogram format
        plt.xlabel("Counter values")
        plt.ylabel("Frequency")
        plt.title("Frequency histogram - " + infoTag)
        default = False

    if plotType == "average_histogram":
        plt.hist(averager(unpklTrace[0]), bins=binNumber, color="orange")
        plt.xlabel("Averaged counter values")
        plt.ylabel("Frequency")
        plt.title(f"Averaged Frequency histogram - " + infoTag)
        default = False

    # If no plot with special format is called, default = true
    if default:
        plt.xlabel(f"Time ({timeUnits})")
        plt.grid("both")

        # If averaged trace is to be plotted, else print plain infoTag
        if average:
             plt.ylabel("Average counter value")
             plt.title("Averaged trace - " + infoTag)
        else:
            plt.ylabel(f"{yAxisUnits}")
            plt.title(infoTag)

    descriptors(averager(unpklTrace[0]), filePath)
    plt.show()

def averager(traceSet):

    # Calculate # of elements in a list within traceSet; create list with that size; traceSet[0] is the first
    # trace in the list and SHOULD have the same size as the averaged trace and all the other traces. 
    size = len(traceSet[0])
    averagedTrace = list(range(size))

    # Iterate through all indexes; average the counter values at each index in all traces in traceSet
    for index in range(size):
        sum = 0
        for trace in traceSet:
            sum += trace[index]
        averagedTrace[index] = sum / len(traceSet)

    return averagedTrace

def descriptors(trace, filePath):

    print()
    print(f"CENTRAL TENDENCIES OF AVERAGED {filePath}")

    # 'trace' must always be one trace, ideally an averaged trace
    print(f"Mean: {round(mean(trace))} ± {round(sem(trace))}")
    print(f"Median: {round(median(trace))}")
    print(f"Standard deviation: {round(stdev(trace))}")
    print()
        
def unpickle(traceFileName, quantity):

    traceFileHandle = open(traceFileName, "rb")
    traceSet = []
    numRuns = 0

    # Call pickle.load multiple times to append multiple traces to the trace set
    # TraceSet is is used by other functions and will always be the first index of the return value of the unpickle function. 
    if quantity == "multi":
        while True:
            try:
                trace, site = load(traceFileHandle) 
                traceSet.append(trace[0])
                numRuns += 1
            except EOFError:
                break
        traceFileHandle.close()
        return traceSet, site, numRuns

    # If only one trace is requested, return numRuns = 1
    trace, site = load(traceFileHandle)
    traceSet.append(trace[0])
    traceFileHandle.close()
    return traceSet, site, 1

if __name__ == "__main__":
    main()
