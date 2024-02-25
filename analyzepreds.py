from readpredsin import predictionPaths
from sklearn.metrics import f1_score
from scipy.stats import pearsonr, sem
from sys import argv, exit
from plotpreds import axis
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def main():

    if len(argv) != 4:
        exit('Usage: python3 plotpreds.py [amount] [plotType] [browser]')

    if argv[1] != 'individual' and argv[1] != 'average':
        exit('You must indicate "individual" or "average"')

    amount = argv[1]
    plotType = argv[2]
    browser = argv[3]
    
    scores = []
    y_errors = []
    avg_scores = []
    avg_y_errors = []

    # Use the list of path lists in readpredsin.py
    for paths in predictionPaths:
        for path in paths:
            f1_scores = []
            arrays = np.load(path)
            true = arrays['true']
            preds = arrays['preds']

            # Calculate F1 scores and append them to f1_scores
            for i in range(len(true)):
                f1_scores.append(f1_score(true[i], preds[i].argmax(axis=1)) * 100)
            scores.append(np.mean(f1_scores))
            y_errors.append(round(sem(f1_scores), 2))

        if amount == 'average':
            avg_scores.append(np.mean(scores))
            avg_y_errors.append(round(sem(scores), 2))
            scores.clear()

    
    binNumber = 16
    histInfoTag = f"Accuracy Frequency distribution - {browser} - Word experiments"
    boxInfoTag = f"Box and whisker plot - {browser} - Word experiments"
    if plotType == 'histogram' and amount == 'individual':
        plt.hist(scores, bins=binNumber, color="orange")
        plt.title(histInfoTag)
        plt.ylabel('Frequency')
        plt.xlabel('F1 score (%)')

    if plotType == 'boxplot' and amount == 'individual':
        plt.boxplot(scores)
        plt.title(boxInfoTag)
        plt.ylabel('F1 score (%)')
        plt.xticks([])
    
    plt.grid('both')
    plt.show()

# Don't get rid of the outliers in the Firefox and Chrome. You can use your results from these word experiments to show that 
# changing word count doesn't really affect accuracy.


if __name__ == "__main__":
    main()
