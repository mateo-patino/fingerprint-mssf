from readpredsin2 import videoPredictionPaths
from sklearn.metrics import f1_score
from scipy.stats import spearmanr, sem
from sys import argv, exit
import matplotlib.pyplot as plt
import numpy as np

# Takes in two command-line arguments: python3 plotpredshist.py [title]

def main():

    if len(argv) != 2:
        exit('Usage: python3 plotpredshist.py [title]')

    titleText = argv[1]

    heights = [] # Bar heights = average score of ten trained models
    error_bars = []
    domainPairs = set()

    # Use the list of path lists in readpredsin.py
    for paths in videoPredictionPaths:
        scores = [] # Initialize new lists each time
        errors = []
        firstItr = True
        for path in paths:
            f1_scores = []
            arrays = np.load(path)
            true = arrays['true']
            preds = arrays['preds']

            if firstItr:
                domainNames = arrays['domains'].tolist()
                domainPairs.add(domainNames[0])
                domainPairs.add(domainNames[1])
                firstItr = False

            # Calculate F1 scores and append them to f1_scores
            for i in range(len(true)):
                f1_scores.append(f1_score(true[i], preds[i].argmax(axis=1)) * 100)
            scores.append(np.mean(f1_scores))
            errors.append(sem(f1_scores)) 

        heights.append(np.mean(scores))
        error_bars.append(np.mean(errors))
    
    # Define categories MANUALLY
    xAxis = ['No video', '1 video (50 kB)', '10 videos (500 kB)']

    # Print order files were opened; this order should ALWAYS match the order in 'xAxis'
    print()
    print(f"First pair: {list(domainPairs)[0]}, {list(domainPairs)[1]}")
    print(f"Second pair: {list(domainPairs)[2]}, {list(domainPairs)[3]}")
    print()

    # Plot a bar chart
    plt.bar(xAxis, heights, label=xAxis, color=['tab:blue', 'tab:orange', 'tab:red'], width=0.45, yerr=error_bars, capsize=5, ec='black',ls='-')
    plt.xlim(-1, 3)
    plt.ylim((0, 110))
    plt.ylabel('F1 score (%)')
    plt.grid(axis='y')
    plt.title(titleText)
    plt.yticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    # ANY WAYS TO IMPLEMENT A P VALUE?

    # Print p-value; this value is recorded in a separate txt.file.
    #x = [0, 50, 500] # Pass in values as kilobyte equivalents (if '1 video' = 50KB, pass in 50)
    #print(f"p = {stats(x, heights)}")

    plt.show()


# Return p-value via the Spearman's r function
def stats(x, y):
    r, p = spearmanr(x, y)
    return r, p
    

if __name__ == '__main__':
    main()