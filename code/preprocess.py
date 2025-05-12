import os
from collections import defaultdict
import shutil
from sklearn.model_selection import train_test_split
import random
import cv2
import os
from pathlib import Path
random.seed(2025)
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  

print(ROOT)
skin_folder = ROOT/"foot-wound-healthy-skin-split"
otherwound_folder = ROOT/"foot-wound-other-wound-split"
def create_dataset(folder, c, test_size=0.2, size=(224,224)):
    data_group = defaultdict(list)
    for root, folder, files in os.walk(folder):
        for f in files:
            f = f.lower()
            if f.endswith("jpg") or f.endswith("png") or f.endswith("jpeg"):
                label = root.split("\\")[-1]
                data_group[label].append(os.path.join(root, f))


    if not os.path.exists("../dataset_{c}"):
        os.makedirs(f"../dataset_{c}/train/healthy-skin", exist_ok=True)
        os.makedirs(f"../dataset_{c}/train/ulcer", exist_ok=True)
        os.makedirs(f"../dataset_{c}/test/healthy-skin", exist_ok=True)
        os.makedirs(f"../dataset_{c}/test/ulcer", exist_ok=True)


    for k, v in data_group.items():

        train_paths, test_paths= train_test_split(v, test_size=test_size, shuffle=True) 
        for i, tr_path in enumerate(train_paths):
            im = cv2.imread(tr_path)
            im = cv2.resize(im, dsize=size)
            cv2.imwrite(f"../dataset_{c}/train/{k}/{i}.jpg", im)
    
        for i, te_path in enumerate(test_paths):
            im = cv2.imread(te_path)
            im = cv2.resize(im, dsize=size)
            cv2.imwrite(f"../dataset_{c}/test/{k}/{i}.jpg", im)


def create_dataset_phase1(folder, c, test_size=0.2, size=(380,224)):
    data_group = defaultdict(list)
    
    for root, folder, files in os.walk(folder):
        for f in files:
            f = f.lower()
            if f.endswith("jpg") or f.endswith("png") or f.endswith("jpeg"):
                label = root.split("\\")[-1]
                path = os.path.join(root, f)
                im = cv2.imread(path)
                h, w, _ = im.shape
                h = max(w, h)
              
                if h < 570:
                    data_group[label].append(path)
                    

    if not os.path.exists("../dataset_{c}"):
        os.makedirs(f"../dataset_{c}/phase1/train/foot-wound", exist_ok=True)
        os.makedirs(f"../dataset_{c}/phase1/train/other-wound", exist_ok=True)
        os.makedirs(f"../dataset_{c}/phase1/test/foot-wound", exist_ok=True)
        os.makedirs(f"../dataset_{c}/phase1/test/other-wound", exist_ok=True)


    for k, v in data_group.items():

        train_paths, test_paths= train_test_split(v, test_size=test_size, shuffle=True) 
        for i, tr_path in enumerate(train_paths):
            im = cv2.imread(tr_path)
            w,h,_ = im.shape
            if w < h:
                im = im.transpose(1,0,2)
            im = cv2.resize(im, dsize=size)
            cv2.imwrite(f"../dataset_{c}/phase1/train/{k}/{i}.jpg", im)
    
        for i, te_path in enumerate(test_paths):
            im = cv2.imread(te_path)
            h, w, _ = im.shape
            if h < w:
                im = im.transpose(1,0,2)
            im = cv2.resize(im, dsize=size)
            cv2.imwrite(f"../dataset_{c}/phase1/test/{k}/{i}.jpg", im)



if __name__ == "__main__":
    create_dataset(skin_folder, c="skin")
    create_dataset_phase1(otherwound_folder, c="other")