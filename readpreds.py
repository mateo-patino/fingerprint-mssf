import numpy as np
from sklearn.metrics import f1_score, top_k_accuracy_score

path = "/Users/mateopatinohasbon/Documents/bigger-fish-main/fingerprint-mssf/chrome_traces/training/predictions/words/20k-100words.npz"
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
print(f"F1 mean: {round(np.mean(f1_scores), 2)}")
print(f"Top-1 mean: {round(np.mean(top1_scores), 2)}")
