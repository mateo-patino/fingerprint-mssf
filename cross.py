from pickle import load
from sys import argv
from statistics import mean, stdev, median
from scipy.stats import sem
import matplotlib.pyplot as plt

#   COMMAND-LINE ARGUMENTS: python3 cross.py trace_length plot_type titleText

def main():

    if len(argv) != 4:
        exit("Missing one command-line argument or too many.")

    # Main parameters

    traceLength = int(argv[1])
    plotType = argv[2]
    titleText = argv[3]
    inputFile = "cross.in"

    timeAxis = range(traceLength * 1000)

    with open(inputFile, "r") as tracePaths:
        paths = tracePaths.readlines()
        # Remove newline characters from the paths
        for index in range(len(paths)):
            paths[index] = paths[index].strip()
    
    # Create set of averaged traces to plot
    averagedTraces = []
    traceLabels = []

    # Store the average trace and its corresponding name on different lists at the same index
    for path in paths:
        traceSet, site, numRuns = unpickle(path)
        averagedTraces.append(averager(traceSet))
        traceLabels.append(formattedName(site))
    
    # Plot traces in averagedTraces list
    if plotType == "line":

        # The average trace and its label share same index
        for index in range(len(averagedTraces)):
            plt.plot(timeAxis, averagedTraces[index], label=traceLabels[index])
    
    # Print descriptors; they help identifying which trace belongs to what browser if different browsers are plotted.
    # Averaged traces and their original pkl file paths are located at the same indexes. 
    for index in range(len(averagedTraces)):
        descriptors(averagedTraces[index], paths[index]) 
    
    # Plot features
    plt.xlabel("Time (ms)")
    plt.ylabel("Averaged counter values")
    plt.title(titleText)
    plt.legend(loc="lower right")
    plt.grid("both")
    plt.show()


def descriptors(trace, filePath):

    print()
    print(filePath)

    # 'trace' must always be one trace, ideally an averaged trace
    print(f"Mean: {round(mean(trace))} ± {round(sem(trace))}")
    print(f"Median: {round(median(trace))}")
    print(f"Standard deviation: {round(stdev(trace))}")
    print()


def averager(traceSet):

    # Calculate # of elements in a list within traceSet; create list with that size; traceSet[0] is the first
    # trace in the list and SHOULD have the same size as the averaged trace and all the other traces in traceSet. 
    size = len(traceSet[0])
    averagedTrace = list(range(size))

    # Iterate through all indexes; average the counter values at each index in all traces in traceSet
    for index in range(size):
        sum = 0
        for trace in traceSet:
            sum += trace[index]
        averagedTrace[index] = sum / len(traceSet)

    return averagedTrace

def formattedName(site):

    # Splits, say "https://cnn.com", by / and
    # then splits cnn.com (last element, hence -1) by . and returns the first element in the resulting list. Works the same for custom trace files.

    return site.split("/")[-1].split(".")[0]


def unpickle(traceFileName):

    traceFileHandle = open(traceFileName, "rb")
    traceSet = []
    numRuns = 0

    # Call pickle.load multiple times to append multiple traces to the trace set
    # TraceSet is extensively used by other functions and will always be the first index of the return value of the unpickle function. 
    while True:
        try:
            trace, site = load(traceFileHandle) 
            traceSet.append(trace[0])
            numRuns += 1
        except EOFError:
            break
    traceFileHandle.close()
    return traceSet, site, numRuns


if __name__ == "__main__":
    main()