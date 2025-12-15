import pickle

with open(r"src\heteromark\benchmark\smac2_training_results.pkl", "rb") as file:
    data = pickle.load(file)

print(data)
