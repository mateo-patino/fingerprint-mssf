from readpredsin import predictionPaths
from sklearn.metrics import f1_score
from scipy.stats import pearsonr, spearmanr, sem
from scipy.optimize import curve_fit
from sys import argv, exit
import matplotlib.pyplot as plt
import numpy as np

# Takes in two command-line arguments: [browser] [plot type]
# To make it easier (or at least faster, maybe not simpler) to pass in the prediction files, I use a Python list
# in readpredsin.py to organize the filenames into a 2d array that's compatible with this code. 

testType = 'spearman' # CHANGE MANUALLY

def main():

    if len(argv) != 3:
        exit('Usage: python3 plotpreds.py [browser] [plotType]')

    plotType = argv[2]
    browser = argv[1]
    scores = []
    y_errors = []

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
            y_errors.append(round(sem(f1_scores), 2)) # I'M NO LONGER PLOTTING ERROR BARS, SO THIS LIST IS UNUSED

    xAxis = axis()
    scores = np.array(scores)
    xAxis = np.array(xAxis)

    # I MANUALLY change these parameters below depending on what plot design I want
    upperBound = 1000 # Maximum difference used
    lowerBound = 100
    automaticTicks = False
    xseq = np.linspace(lowerBound, upperBound, 500)

    # Fit y = B + Alog(x)
    if plotType == 'log':

        A, B = np.polyfit(np.log(xAxis), scores, 1)
        reg_y_axis = np.log(xseq) * A
        reg_y_axis = reg_y_axis + B

    if plotType == 'line':

        slope, intercept = np.polyfit(xAxis, scores, deg=1)
        reg_y_axis = xseq * slope
        reg_y_axis = reg_y_axis + intercept

    if plotType == 'logistic':

        popt, pcov = curve_fit(logistic_function, xAxis, scores, bounds=(0, [100.0, 1.0, 100.0]))
        L, k, x0 = popt
        reg_y_axis = logistic_function(xseq, L, k, x0)

    if plotType != 'none': plt.plot(xseq, reg_y_axis, color='red', lw=2)

    plt.scatter(xAxis, scores, s=50, color='black')

    print(f'Length of "scores": {len(scores)}')
    rcoef, pvalue = stats(xAxis, scores)

    print(f'p = {pvalue}\nr = {rcoef}')    

    # Set text box up to include r and p-value
    values = (f'p = {pvalue}\n'
             f'{testType.capitalize()}\'s r = {rcoef}')
    bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5, pad=0.5)

    # Plot information
    plt.xlabel('Image count difference')
    plt.ylabel('F1 score (%)')
    plt.grid(axis='y')
    plt.ylim((0, 110))
    plt.xlim((0, 1100)) # This xlim is also changed MANUALLY before each run
    plt.title(f'F1 accuracy score vs. Image count difference - {browser.capitalize()} - 4 models per difference level')
    plt.yticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    # Add stats box; location is also MANUALLY changed to fix xaxis scale
    plt.text(upperBound - 205, 11, values, fontsize=12, bbox=bbox,
            horizontalalignment='left')
    
    # Set x-axis ticks
    if not automaticTicks: plt.xticks(x_ticks())

    plt.show()    

def logistic_function(x, L, k, x0):
   return L / (1 + np.exp(-k * (x - x0)))

# I MANUALLY select the ticks and axis I want to use before running this script, so I just
# change the return value of the two functions below depending on what I'm plotting. 

def x_ticks():

    xTicks1 = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    xTicks2 = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    xTicks3 = [3000, 6000, 9000, 12000, 15000, 18000, 21000, 24000, 27000, 30000]

    xTrial3 = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

    xTicks4 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    xTicks5 = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    
    return xTicks2
    
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
        930, 1000, 1070, 1140,
        1930, 2000, 2070, 2140,
        2930, 3000, 3070, 3140,
        3930, 4000, 4070, 4140,
        4930, 5000, 5070, 5140,
        5930, 6000, 6070, 6140,
        6930, 7000, 7070, 7140,
        7930, 8000, 8070, 8140,
        8930, 9000, 9070, 9140,
        9930, 10000, 10070, 10140
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

    xAxis8 = [0.85, 0.95, 1.05, 1.15,
              1.85, 1.95, 2.05, 2.15,
              2.85, 2.95, 3.05, 3.15,
              3.85, 3.95, 4.05, 4.15,
              4.85, 4.95, 5.05, 5.15,
              5.85, 5.95, 6.05, 6.15,
              6.85, 6.95, 7.05, 7.15,
              7.85, 7.95, 8.05, 8.15,
              8.85, 8.95, 9.05, 9.15,
              9.85, 9.95, 10.05, 10.15,
    ]

    xAxis9 = [5.85, 5.95, 6.05, 6.15,
          10.85, 10.95, 11.05, 11.15,
          15.85, 15.95, 16.05, 16.15,
          20.85, 20.95, 21.05, 21.15,
          25.85, 25.95, 26.05, 26.15,
          30.85, 30.95, 31.05, 31.15,
          35.85, 35.95, 36.05, 36.15,
          40.85, 40.95, 41.05, 41.15,
          45.85, 45.95, 46.05, 46.15,
          50.85, 50.95, 51.05, 51.15
    ]
    
    return xAxis7

# returns list with p-values and Pearson's r
def stats(x, y):

    if testType == 'spearman':
        r, p = spearmanr(x, y)

    elif testType == 'pearson':
        r, p = pearsonr(x, y)

    # I round p to different values depending on how small it is
    return round(r, 2), round(p, 2)


if __name__ == "__main__":
    main()