from sys import argv
from pickle import load
import matplotlib.pyplot as plt
def main():

    file1 = argv[1]
    file2 = argv[2]

    trace1 = unpickle(file1)[0][0]
    trace2 = unpickle(file2)[0][0]

    timeAxis = range(15000)

    plt.plot(timeAxis, trace1, color="red")
    plt.plot(timeAxis, trace2, color="blue")
    plt.plot(timeAxis, substractTraces(trace1, trace2), color="orange")
    plt.title(" Example: subtraction of 10-second cnn.com trace from Firefox (blue) and Chrome (red).")
    plt.ylabel("Counter values")
    plt.xlabel("Time (ms)")
    plt.grid("both")
    plt.show()

    print(substractTraces(trace1, trace2))

def substractTraces(trace1, trace2):

    size = len(trace1) 
    resultTrace = list(range(size))
    for index in range(size): 
        resultTrace[index] = trace2[index] - trace1[index]

    return resultTrace


def unpickle(traceFileName):

    traceFileHandle = open(traceFileName, "rb")
    traceSet = []
    numRuns = 0

    # Call pickle.load multiple times to append multiple traces to the trace set
    # TraceSet is is used by other functions and will always be the first index of the return value of the unpickle function. 
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