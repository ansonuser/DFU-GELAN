# %%

import cv2
import os
from collections import defaultdict
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
print(os.getcwd())

# %%

folder_name = "../foot-wound-other-wound-split"
file_paths = []
group = defaultdict(list)
for root, dir, files in os.walk(folder_name):
    
    for f in files:
        if f.endswith("jpg") or f.endswith("png"):
            key = root.split("\\")[-1]
            group[key].append(os.path.join(root, f))
            file_paths.append(os.path.join(root, f))



# print(len(file_paths))
for k, v in group.items():
    print(k)
    print(len(v))

# %%



# %%
key = "foot-wound" # 45, 1.755 | 510, 1.8735
key = "other-wound" # 598, 1.69 | 85, 2.166
datasize_collections = defaultdict(list)
ratio_sum = 0
bin_width = 100
count = 0
file_paths = group[key]
for f in file_paths:
    img = cv2.imread(f)
    h, w, c = img.shape
    w = w//bin_width
    h = h//bin_width
    if h < w:
        h, w = w, h
        
    shape = (h, w)    
    # if w >= 3:
    ratio_sum += shape[0]/shape[1]
    count += 1
    datasize_collections[shape].append(f)

ratio_hat = ratio_sum/count
print("valid count:", count)
print("ratio:", ratio_hat)

# %%
import numpy as np

counter = {}
w_sum = 0
h_sum = 0
for k, v in datasize_collections.items():
    w_sum += k[0]*k[1]*len(v)
    counter[k] = len(v)
w_hat = np.sqrt(w_sum/(ratio_hat*len(file_paths)))
h_hat = w_hat*ratio_hat
print(f"w = {w_hat}, h = {h_hat}")


# %%
datasize_collections

# %%
information_changes = []
issue_files = []
for k, v in datasize_collections.items():
    w_min = min(w_hat, k[0])
    h_min = min(h_hat, k[1])
    intersection = w_min*h_min
    information_changes += [1-intersection/(k[0]*k[1])]*len(v)
    if (1-intersection/(k[0]*k[1])) > .5:
        issue_files += v
    
print(f"# of sample:{len(information_changes)}")
# %%
plt.hist(information_changes)
# %%
df = pd.DataFrame.from_dict(counter, orient='index', columns=['count'])
df = df.reset_index()

df["x"] = df["index"].apply(lambda x: x[0])
df["y"] = df["index"].apply(lambda x: x[1])
heatmap_data = df.pivot_table(index='y', columns='x', values='count')
heatmap_data.fillna(0, inplace=True)
# %%
sns.heatmap(heatmap_data)

plt.xlabel("X")
plt.ylabel("Y")
plt.title("2D Count Heatmap")
plt.show()

# %%
df["total_bins"] = df["x"]*df["y"]
bin_count = df.groupby("total_bins")["count"].sum().reset_index()
print(bin_count)
# %%
cumsum = np.cumsum(bin_count["count"])
cdf = cumsum/cumsum.values[-1]

# %%
plt.plot(bin_count["total_bins"][:-10], cdf.values[:-10])
# %%
imgs = []
for file in issue_files:
    imgs.append(cv2.imread(file))


# %%
def show_cv2_images(images, rows=4, cols=4, figsize=(12, 12)):
    assert len(images) >= rows * cols, "Not enough images to fill the grid."

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    for i, ax in enumerate(axes.flat):
        img = images[i]
        print(img.shape)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
# %%



    
