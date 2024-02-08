import numpy as np
from sklearn.metrics import f1_score, top_k_accuracy_score

data = np.load("/Users/mateopatinohasbon/Documents/bigger-fish-main/10k-100-words.npz")

true = data["true"]
preds = data["preds"]

f1_scores = []
top1_scores = []

for i in range(len(true)):
    f1_scores.append(f1_score(true[i], preds[i].argmax(axis=1)) * 100)
    top1_scores.append(top_k_accuracy_score(true[i], preds[i][:, 1], k=1) * 100)

print(f"F1 scores: {f1_scores}")
print(f"Top1 scores: {top1_scores}")
print()
print(f"F1 mean: {round(np.mean(f1_scores), 2)}")
print(f"Top1 mean: {round(np.mean(top1_scores), 2)}")
