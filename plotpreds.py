from readpredsin import predictionPaths
from sklearn.metrics import f1_score
from scipy.stats import ttest_1samp, sem
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# This script gives a scatter plot with the mean F1 scores for the number of models trained per pair of pages

def main():
    
    categories = []
    scores = []

    # Use the list of path lists in readpredsin.py
    for paths in predictionPaths:
        f1_scores = []
        for path in paths:
            arrays = np.load(path)
            true = arrays['true']
            preds = arrays['preds']

            # Calculate F1 scores and append them to f1_scores
            for i in range(len(true)):
                f1_scores.append(f1_score(true[i], preds[i].argmax(axis=1)) * 100)
        
        scores.append(np.mean(f1_scores))
        categories.append(label(paths[0]))

    data = {
        'Word count difference': categories,
        'F1 score': scores
    }

    sns.stripplot(x='Word count difference', y='F1 score', data=data, order=data['Word count difference'], jitter=True, size=5, color='green')

    plt.xlabel('Word count difference')
    plt.ylabel('F1 score (%)')
    plt.grid('both')
    plt.ylim((0, 100))
    plt.title('Stripplot with Arbitrary Labels on Horizontal Axis')

    plt.show()    
    

def label(path):
    l = ""
    tmp0 = path.split("/")[-1]

    # The first characters will always be numbers; stop when reaching something else
    for c in tmp0:
        if c.isdigit():
            l = l + c
        else:
            if c == 'k':
                return l + 'k'
            break
    
    # The difference between two pages (in the hundreds scale) is obtained by subtracting their word counts; base word count is 100
    return str(int(l) - 100)


if __name__ == "__main__":
    main()