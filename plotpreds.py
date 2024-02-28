from readpredsin import predictionPaths
from sklearn.metrics import f1_score
from scipy.stats import pearsonr, spearmanr, sem
from sys import argv, exit
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# YOU CAN DO ONES, FIVES AND TENS WITH THE VIDEO PAGES YOU CREATED. FOR TENS, YOU SET THE BASE TO BE 1 AND USE 11, 21, 31, 41, AND 51

# This script gives a scatter plot with the mean F1 scores for the number of models trained per pair of pages
# This script takes in two command-line arguments: "average" or "individual" and browser name. "average" plots the average F1 of the four
# trained models; "individual" plots the individual F1 scores for each of the four trained models. 

testType = 'spearman' # CHANGE MANUALLY

def main():

    if len(argv) != 3:
        exit('Usage: python3 plotpreds.py [amount] [browser]')

    if argv[1] != 'individual' and argv[1] != 'average':
        exit('You must indicate "individual" or "average"')
    
    scores = []
    avg_scores = []


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

        if argv[1] == 'average':
            avg_scores.append(np.mean(scores))
            scores.clear()

    xAxis = axis()
    if argv[1] == 'average':

        # Compute slope and y-intercept for regression line and plot error bars
        slope, intercept = np.polyfit(xAxis, avg_scores, deg=1)
        plt.scatter(xAxis, avg_scores, s=50, color='black')

        # Compute r and p-value
        rcoef, pvalue = stats(xAxis, avg_scores)
        print(f'Length of "avg_scores": {len(avg_scores)}')
        print('The length of "avg_score" is relevant because it is the sample used\nto calculate Pearson\'s r and p-values. ')

    elif argv[1] == 'individual':

        slope, intercept = np.polyfit(xAxis, scores, deg=1)
        plt.scatter(xAxis, scores, s=50, color='black')

        print(f'Length of "scores": {len(scores)}')
        print('The length of "score" is relevant because it is the sample used \n to calculate r and p-values. ')
        rcoef, pvalue = stats(xAxis, scores)
    

    print(f'p = {pvalue}\nr = {rcoef}')    
    
    # Compute regression line
    upperBound = 10000 # Maximum difference used; CHANGED MANUALLY
    lowerBound = 1000
    xseq = np.linspace(lowerBound, upperBound, 50)
    reg_y_axis = xseq * slope
    reg_y_axis = reg_y_axis + intercept
    plt.plot(xseq, reg_y_axis, color='red', lw=2)

    # Set text box up to include r and p-value
    values = (f'p = {pvalue}\n'
             f'{testType.capitalize()}\'s r = {rcoef}')
    bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5, pad=0.5)

    # Plot information
    plt.xlabel('Image count difference')
    plt.ylabel('F1 score (%)')
    plt.grid(axis='y')
    plt.ylim((0, 110))
    plt.xlim((0, 11000))
    plt.title(f'F1 accuracy score vs. image count difference - {argv[2].capitalize()} - 4 models per difference level')
    plt.yticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    # Add stats box
    plt.text(upperBound - 4000, 11, values, fontsize=11, bbox=bbox,
            horizontalalignment='left')
    
    # Set x-axis ticks
    automaticTicks = False
    if not automaticTicks:
        plt.xticks(x_ticks())

    plt.show()    

def x_ticks():

    # I MANUALLY SELECT WHAT TICKS I WANT TO RETURN BEFORE RUNNING THE SCRIPT

    xTicks1 = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    xTicks2 = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    xTicks3 = [3000, 6000, 9000, 12000, 15000, 18000, 21000, 24000, 27000, 30000]

    xTrial3 = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    
    return xTrial3
    
def axis():

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

    xAxis2 = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    xAxis3 = [
        990, 1000, 1010, 1020,
        1990, 2000, 2010, 2020,
        2990, 3000, 3010, 3020,
        3990, 4000, 4010, 4020,
        4990, 5000, 5010, 5020,
        5990, 6000, 6010, 6020,
        6990, 7000, 7010, 7020,
        7990, 8000, 8010, 8020,
        8990, 9000, 9010, 9020,
        9990, 10000, 10010, 10020,
    ]

    xAxis4 = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000,
              21000, 22000, 23000, 24000, 25000, 26000, 27000, 28000, 29000, 30000]
    

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


    # Pic experiments, tens scale
    xAxis6 = [
        9, 10, 11, 12,
        19, 20, 21, 22,
        29, 30, 31, 32,
        39, 40, 41, 42,
        49, 50, 51, 52,
        59, 60, 61, 62,
        69, 70, 71, 72,
        79, 80, 81, 82,
        89, 90, 91, 92,
        99, 100, 101, 102
    ]

    # Pic experiment, hundreds scale
    xAxis7 = [
        98, 99, 100, 101,
        198, 199, 200, 201,
        298, 299, 300, 301,
        398, 399, 400, 401,
        498, 499, 500, 501,
        598, 599, 600, 601,
        698, 699, 700, 701,
        798, 799, 800, 801,
        898, 899, 900, 901,
        998, 999, 1000, 1001
    ]

    
    return xAxis3

# returns list with p-values and Pearson's r
def stats(x, y):

    if testType == 'spearman':
        r, p = spearmanr(x, y)

    elif testType == 'pearson':
        r, p = pearsonr(x, y)

    return round(r, 2), round(p, 2)


if __name__ == "__main__":
    main()