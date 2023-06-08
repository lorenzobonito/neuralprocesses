import pickle

with open("datasets.pickle","rb") as f:
    datasets = pickle.load(f)

with open("datasets_joint.pickle","rb") as f:
    datasets_joint = pickle.load(f)

print(datasets["contexts"][0])
print(datasets_joint[0]["contexts"][0])