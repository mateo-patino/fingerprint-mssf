from readpredsin import predictionPaths
from sklearn.metrics import f1_score
from scipy.stats import pearsonr, spearmanr, sem
from sys import argv, exit
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# This script gives a scatter plot with the mean F1 scores for the number of models trained per pair of pages
# This script takes in two command-line arguments: "average" or "individual" and browser name. "average" plots the average F1 of the four
# trained models; "individual" plots the individual F1 scores for each of the four trained models. 

testType = 'pearson' # CHANGE MANUALLY

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

    errbarcolor = 'gray'
    xAxis = axis()
    if argv[1] == 'average':

        # Compute slope and y-intercept for regression line and plot error bars
        slope, intercept = np.polyfit(xAxis, avg_scores, deg=1)
        plt.scatter(xAxis, avg_scores, s=50, color='dodgerblue')
        plt.errorbar(xAxis, avg_scores, yerr=avg_y_errors, ecolor=errbarcolor, capsize=4, ls='None')

        # Compute r and p-value
        rcoef, pvalue = stats(xAxis, avg_scores)
        print(f'Length of "avg_scores": {len(avg_scores)}')
        print('The length of "avg_score" is relevant because it is the sample used\nto calculate Pearson\'s r and p-values. ')

    elif argv[1] == 'individual':

        slope, intercept = np.polyfit(xAxis, scores, deg=1)
        plt.scatter(xAxis, scores, s=50, color='dodgerblue')
        plt.errorbar(xAxis, scores, yerr=y_errors, ecolor=errbarcolor, capsize=4, ls='None')

        print(f'Length of "scores": {len(scores)}')
        print('The length of "score" is relevant because it is the sample used \n to calculate Pearson\'s r and p-values. ')
        rcoef, pvalue = stats(xAxis, scores)
    

    print(f'p = {pvalue}\nr = {rcoef}')    
    
    # Compute regression line
    upperBound = 30000 # Maximum difference used
    lowerBound = 100
    xseq = np.linspace(lowerBound, upperBound, 50)
    reg_y_axis = xseq * slope
    reg_y_axis = reg_y_axis + intercept
    plt.plot(xseq, reg_y_axis, color='red', lw=2)

    # Set text box up to include r and p-value
    values = (f'p = {pvalue}\n'
             f'{testType.capitalize()} r = {rcoef}')
    bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)

    # Plot information
    plt.xlabel('Word count difference')
    plt.ylabel('F1 score (%)')
    plt.grid('both')
    plt.ylim((0, 120))
    plt.title(f'F1 accuracy score vs. word count difference - {argv[2].capitalize()} - 4 models per difference level')
    plt.yticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    # Add stats box
    plt.text(upperBound - 4000, 12, values, fontsize=10, bbox=bbox,
            horizontalalignment='left')
    
    # Set x-axis ticks
    automaticTicks = False
    if not automaticTicks:
        plt.xticks(x_ticks())

    plt.show()    

def x_ticks():

    # xTicks1 is for individual AND average F1 scores for hundreds Firefox
    xTicks1 = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    
    # xTicks2 is for individual AND average F1 scores for thousands Firefox
    xTicks2 = [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000]

    # xTicks3 is for individual AND average F1 scores for thousands up to 30K Chrome
    xTicks3 = [3000, 6000, 9000, 12000, 15000, 18000, 21000, 24000, 27000, 30000]

    xTrial2 = [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000]

    xTrial3 = [1000, 2000, 3000, 4000, 5000, 6000, 7000]
    
    return xTicks3
    
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

    # xAxis3 is for thousands experiment in Firefox four models per pair
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
        9995.5, 10000, 10005.5, 10010,
        10995.5, 11000, 11005.5, 11010,
        11995.5, 12000, 12005.5, 12010,
        12995.5, 13000, 13005.5, 13010,
        13995.5, 14000, 14005.5, 14010,
        14995.5, 15000, 15005.5, 15010,
        15995.5, 16000, 16005.5, 16010,
        16995.5, 17000, 17005.5, 17010,
        17995.5, 18000, 18005.5, 18010,
        18995.5, 19000, 19005.5, 19010,
        19995.5, 20000, 20005.5, 20010
    ]

    # xAxis4 is for thousands experiment in Chrome, ind, averaged
    xAxis4 = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000,
              21000, 22000, 23000, 24000, 25000, 26000, 27000, 28000, 29000, 30000]
    
    # xAxis5 is for Chrome individual up to 30k
    xAxis5 = [
        995, 1000, 1005.5, 1010,
        1995.5, 2000, 2005.5, 2010,
        2995.5, 3000, 3005.5, 3010,
        3995.5, 4000, 4005.5, 4010,
        4995.5, 5000, 5005.5, 5010,
        5995.5, 6000, 6005.5, 6010,
        6995.5, 7000, 7005.5, 7010,
        7995.5, 8000, 8005.5, 8010,
        8995.5, 9000, 9005.5, 9010,
        9995.5, 10000, 10005.5, 10010,
        10995.5, 11000, 11005.5, 11010,
        11995.5, 12000, 12005.5, 12010,
        12995.5, 13000, 13005.5, 13010,
        13995.5, 14000, 14005.5, 14010,
        14995.5, 15000, 15005.5, 15010,
        15995.5, 16000, 16005.5, 16010,
        16995.5, 17000, 17005.5, 17010,
        17995.5, 18000, 18005.5, 18010,
        18995.5, 19000, 19005.5, 19010,
        19995.5, 20000, 20005.5, 20010,
        20995.5, 21000, 21005.5, 21010,
        21995.5, 22000, 22005.5, 22010,
        22995.5, 23000, 23005.5, 23010,
        23995.5, 24000, 24005.5, 24010,
        24995.5, 25000, 25005.5, 25010,
        25995.5, 26000, 26005.5, 26010,
        26995.5, 27000, 27005.5, 27010,
        27995.5, 28000, 28005.5, 28010,
        28995.5, 29000, 29005.5, 29010,
        29995.5, 30000, 30005.5, 30010
    ]

    xTrial2 = [
        995, 1000, 1005.5, 1010,
        1995.5, 2000, 2005.5, 2010,
        2995.5, 3000, 3005.5, 3010,
        3995.5, 4000, 4005.5, 4010,
        4995.5, 5000, 5005.5, 5010,
        5995.5, 6000, 6005.5, 6010,
        6995.5, 7000, 7005.5, 7010,
        7995.5, 8000, 8005.5, 8010,
        8995.5, 9000, 9005.5, 9010,
        9995.5, 10000, 10005.5, 10010,
        10995.5, 11000, 11005.5, 11010,
        11995.5, 12000, 12005.5, 12010,
        12995.5, 13000, 13005.5, 13010,
        13995.5, 14000, 14005.5, 14010
        ]
    
    xTrial3 = [
        995, 1000, 1005.5, 1010,
        1995.5, 2000, 2005.5, 2010,
        2995.5, 3000, 3005.5, 3010,
        3995.5, 4000, 4005.5, 4010,
        4995.5, 5000, 5005.5, 5010,
        5995.5, 6000, 6005.5, 6010,
        6995.5, 7000, 7005.5, 7010,
    ]
    
    return xAxis5

# returns list with p-values and Pearson's r
def stats(x, y):

    if testType == 'spearman':
        r, p = spearmanr(x, y)

    elif testType == 'pearson':
        r, p = pearsonr(x, y)

    return round(r, 2), round(p, 7)


if __name__ == "__main__":
    main()