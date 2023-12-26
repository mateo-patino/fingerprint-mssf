from pickle import load

with open("chrome_traces/15s/50runs/localhost_8000_nothing.html.pkl", "rb") as file:
    trace = load(file)
    print(trace)

# "https://cnn.com"
# "http://localhost:8000/nothing.html"
    