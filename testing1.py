# Testing a script that iterates through the pickle trace file until it reaches the end of the file (EOFError). 

from sys import argv, exit
import matplotlib.pyplot as plt
import pickle


def main():

    # FILES, THESE ARE THE Y-AXIS
    traceSet = unpickle("chrome_traces/cnn.com.pkl", multiple=True)[0]
    time_axis = range(10000)
    for t in traceSet:
        plt.plot(time_axis, t, lw=1)

    plt.title("Chrome- cnn.com - 10 seconds - 10 runs")
    plt.xlabel("Time (ms)")
    plt.ylabel("Counter values")
    plt.show()


def unpickle(traceFileName, multiple=False):

    if multiple:
        traceFileHandle = open(traceFileName, "rb")
        traceSet = []
        while True:
            try:
                trace, site = pickle.load(traceFileHandle) 
                traceSet.append(trace[0])
            except EOFError:
                break
        traceFileHandle.close()
        return traceSet, site

    with open(traceFileName, "rb") as traceFileHandle:
        return pickle.load(traceFileHandle)

if __name__ == "__main__":
    main()