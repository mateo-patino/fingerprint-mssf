from numpy import load

path = "/Users/mateopatinohasbon/Documents/bigger-fish-main/fingerprint-mssf/chrome_traces/training/predictions/vids/novid-index-chrome//novid-index1.npz"

data = load(path)

print(data['domains'])