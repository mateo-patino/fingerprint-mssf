import numpy as np
from sklearn.metrics import f1_score, top_k_accuracy_score
from scipy.stats import ttest_1samp, sem

prefix = "/Users/mateopatinohasbon/Documents/bigger-fish-main/fingerprint-mssf/chrome_traces/training/predictions/words/"
paths = [
    f"{prefix}200-100words.npz",
    f"{prefix}300-100words.npz",
    f"{prefix}400-100words.npz",
    f"{prefix}500-100words.npz",
    f"{prefix}10k-100words.npz",
    f"{prefix}40k-100words.npz",
    f"{prefix}100k-100words.npz",
    f"{prefix}200k-100words.npz",
    f"{prefix}400k-100words.npz",
]
for path in paths:
    data = np.load(path)
    true = data["true"]
    preds = data["preds"]

    f1_scores = []
    top1_scores = []

    for i in range(len(true)):
        f1_scores.append(f1_score(true[i], preds[i].argmax(axis=1)) * 100)
        top1_scores.append(top_k_accuracy_score(true[i], preds[i][:, 1], k=1) * 100)

    browser = "FIREFOX"
    title = f"{browser} - F1 AND TOP-1 SCORES FOR {path.split('/')[-1]}"
    print()
    print("----------------------------------------------------------")
    print()
    print(title)
    print(f"F1 mean: {round(np.mean(f1_scores), 2)} ± {round(sem(f1_scores), 1)}")
    print(f"Top-1 mean: {round(np.mean(top1_scores), 2)} ± {round(sem(top1_scores), 1)}")
    print()
    f1tTest = ttest_1samp(f1_scores, 0.5)
    top1tTest = ttest_1samp(top1_scores, 0.5)

    print(f"F1 p: {f1tTest.pvalue}")
    print(f"F1 95% CI {round(f1tTest.confidence_interval(confidence_level=0.95)[0], 2), round(f1tTest.confidence_interval(confidence_level=0.95)[1], 2)}")
    print()
    print(f"Top-1 p: {top1tTest.pvalue}")
    print(f"Top-1 95% CI {round(top1tTest.confidence_interval(confidence_level=0.95)[0], 2), round(top1tTest.confidence_interval(confidence_level=0.95)[1], 2)}")
    
