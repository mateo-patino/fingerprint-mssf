from pickle import load

name = "/Users/mateopatinohasbon/Documents/bigger-fish-main/fingerprint-mssf/firefox_traces/training/5s/100r/localhost_8000_200words.html.pkl"

traces = list()
with open(name, "rb") as f:
    while True:
        try:
            trace, label = load(f)
            traces.append(trace)
        except EOFError:
            break

print(len(traces))