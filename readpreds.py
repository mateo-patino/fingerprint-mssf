import numpy as np
from sklearn.metrics import f1_score, top_k_accuracy_score
from scipy.stats import ttest_1samp, sem

prefix = "/Users/mateopatinohasbon/Documents/bigger-fish-main/fingerprint-mssf/chrome_traces/training/predictions/words/"
paths = [
    f"{prefix}400-100words.npz",
    f"{prefix}10k-100words.npz",
    f"{prefix}40k-100words.npz",
    f"{prefix}100k-100words.npz",
    f"{prefix}200k-100words.npz",
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

    browser = "CHROME"
    title = f"{browser} - F1 AND TOP-1 SCORES FOR {path.split('/')[-1]}"
    print(title)
    print(f"F1 scores: {f1_scores}")
    print(f"Top-1 scores: {top1_scores}")
    print(f"F1 mean: {round(np.mean(f1_scores), 2)} ± {round(sem(f1_scores), 1)}")
    print(f"Top-1 mean: {round(np.mean(top1_scores), 2)} ± {round(sem(top1_scores), 1)}")

    print(f"F1 p: {ttest_1samp(f1_scores, 0.5).pvalue}")
    print(f"Top-1 p: {ttest_1samp(top1_scores, 0.5).pvalue}")

    f1confidenceInt = ttest_1samp(f1_scores, 0.5).confidence_interval(confidence_level=0.95)
    top1confidenceInt = ttest_1samp(top1_scores, 0.5).confidence_interval(confidence_level=0.95)
    print(f"F1 95% CI {round(f1confidenceInt[0], 2), round(f1confidenceInt[1], 2)}")
    print(f"Top-1 95% CI {round(top1confidenceInt[0], 2), round(top1confidenceInt[1], 2)}")
    print()
