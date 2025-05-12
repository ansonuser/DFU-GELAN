from sklearn.linear_model import LogisticRegression
import os
import cv2 
from collections import defaultdict
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score
import random
random.seed(2025)

clf = LogisticRegression()

folder_name = "../foot-wound-other-wound-split"
file_paths = []
group = defaultdict(list)
for root, dir, files in os.walk(folder_name):
    for f in files:
        if f.endswith("jpg") or f.endswith("png"):
            key = root.split("\\")[-1]
            img = cv2.imread(os.path.join(root, f))
            h, w, c = img.shape
            group[key].append([h, w, h/w])

rows = [] 
for k, vs in group.items():
    for v in vs:
        if k == "foot-wound":
            v = v + [0]
            rows.append(v)
        else:
            v = v + [1]
            rows.append(v)

df = pd.DataFrame(rows, columns=["h", "w", "ratio", "target"])
test_size = 0.2
tr_idx, te_idx = train_test_split(range(len(df)), test_size=test_size, shuffle=True, stratify=df["target"], random_state=2025)

clf.fit(df.iloc[tr_idx,:-1], df.iloc[tr_idx,-1])
res = clf.predict(df.iloc[te_idx,:-1])
cm = confusion_matrix(res, df.iloc[te_idx, -1])
acc = accuracy_score(res, df.iloc[te_idx, -1])
recall = recall_score(res, df.iloc[te_idx, -1], average='binary')  

print("Confusion matrix:")
print(cm)
print("Acc:")
print(acc)
print("Recall:")
print(recall)
# print("acc:", (res == df.iloc[te_idx, -1]).mean())
