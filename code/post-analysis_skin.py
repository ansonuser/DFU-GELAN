import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score

def extract_label(x):
    split_x = x.split("\\")
    label = split_x[-3] + '_' + split_x[-2]
    return label

path = "../output/output.txt"
with open(path, "r") as f:
    df = f.readlines()

rows = []

for r in df:
    r = r.rstrip().split(" ")
    r = [k for i, k in enumerate(r) if i in [2, 4, 5]]
    rows.append(r)


df = pd.DataFrame(rows, columns=["filename", "prediction", "predict_prob"])

df["tag"] = df["filename"].apply(lambda x: extract_label(x))
df["label"] = df["tag"].apply(lambda x: x.split("_")[1]).map({"healthy-skin":0, "ulcer":1})
df["classification"] = df["prediction"].apply(lambda x: int(x.split("_")[1])) 

incorrect_indx = df["label"] != df["classification"]
print(f"Test in sample size:{len(df)}")
num_of_pos = (df["label"] == 1).sum()
num_of_neg = (df["label"] == 0).sum()
print("Healthy size:", num_of_pos)
print("Ulcear size:", num_of_neg)
print("Fail to classify the following correctly: ")
print(df[incorrect_indx]["filename"].values)
print("They were classified as:")
print(df[incorrect_indx]["classification"].values)
cm = confusion_matrix(df["label"], df["classification"])
acc = accuracy_score(df["label"], df["classification"])
recall = recall_score(df["label"], df["classification"], average='binary')  

print("Confusion matrix:")
print(cm)
print("Acc:")
print(acc)
print("Recall:")
print(recall)