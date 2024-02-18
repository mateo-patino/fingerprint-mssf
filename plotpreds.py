from readpredsin import predictionPaths
from sklearn.metrics import f1_score
from scipy.stats import ttest_1samp, sem
from sys import argv, exit
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# This script gives a scatter plot with the mean F1 scores for the number of models trained per pair of pages
# This script takes in two command-line arguments: "average" or "individual" and browser name. "average" plots the average F1 of the four
# trained models; "individual" plots the individual F1 scores for each of the four trained models. 

def main():

    if len(argv) != 3:
        exit('Usage: python3 plotpreds.py [amount] [browser]')

    if argv[1] != 'individual' and argv[1] != 'average':
        exit('You must indicate "individual" or "average"')
    
    
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

        if argv[1] == 'average':
            avg_scores.append(np.mean(scores))
            avg_y_errors.append(round(sem(scores), 2))
            scores.clear()

    xAxis = axis()
    if argv[1] == 'average':

        # Compute slope and y-intercept for regression line and plot error bars
        slope, intercept = np.polyfit(xAxis, avg_scores, deg=1)
        plt.errorbar(xAxis, avg_scores, yerr=avg_y_errors, fmt='o', ecolor='gray', capsize=4, color='red')

    elif argv[1] == 'individual':
        slope, intercept = np.polyfit(xAxis, scores, deg=1)
        plt.errorbar(xAxis, scores, yerr=y_errors, fmt='o', ecolor='gray', capsize=4, color='red')

    plt.style.use('fast')
    
    # Compute regression line
    upperBound = 1000 # Maximum difference used
    lowerBound = 100
    xseq = np.linspace(lowerBound, upperBound, 50)
    reg_y_axis = xseq * slope
    reg_y_axis = reg_y_axis + intercept
    plt.plot(xseq, reg_y_axis, color='blue', lw=2)

    # Plot information
    plt.xlabel('Word count difference')
    plt.ylabel('F1 score (%)')
    plt.grid('both')
    plt.ylim((0, 100))
    plt.title(f'F1 accuracy score vs. word count difference - {argv[2].capitalize()} - 4 models per difference level')
    plt.yticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    automaticTicks = False

    if not automaticTicks:
        plt.xticks(x_ticks())

    plt.show()    

def x_ticks():

    # xTicks1 is for individual AND average F1 scores for hundreds Firefox
    xTicks1 = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    
    # xTicks2 is for individual AND average F1 scores for thousands Chrome
    xTicks2 = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    
    return xTicks1
    
def axis():

    # xAxis1 is for individual F1 scores for hundreds Firefox
    xAxis1 = [
        95, 100, 105.5, 101,
        195.5, 200, 205.5, 210,
        295.5, 300, 305.5, 310,
        395.5, 400, 405.5, 410,
        495.5, 500, 505.5, 510,
        595.5, 600, 605.5, 610,
        695.5, 700, 705.5, 710,
        795.5, 800, 805.5, 810,
        895.5, 900, 905.5, 910,
        995.5, 1000, 1005.5, 1010
    ]

    # xAxis2 is for averaged F1 scores for hundreds Firefox
    xAxis2 = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    # xAxis3 is for thousands experiment in Chrome, ind, four models per pair
    xAxis3 = [
        995, 1000, 1005.5, 1010,
        1995.5, 2000, 2005.5, 2010,
        2995.5, 3000, 3005.5, 3010,
        3995.5, 4000, 4005.5, 4010,
        4995.5, 5000, 5005.5, 5010,
        5995.5, 6000, 6005.5, 6010,
        6995.5, 7000, 7005.5, 7010,
        7995.5, 8000, 8005.5, 8010,
        8995.5, 9000, 9005.5, 9010,
        9995.5, 10000, 10005.5, 10010
    ]

    xAxis4 = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

    return xAxis2

# 1) Reach 20k in the chrome plot; you already have the pkl data, now you got to run it through Colab
# 2) You'll repeat the Firefox experiments but using the increments of 1000 you used for Chrome
# 3) Do what Jack suggests about loading the images right after the page loads and using larger increments


if __name__ == "__main__":
    main()