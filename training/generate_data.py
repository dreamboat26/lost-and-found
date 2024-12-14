import glob
import os
import string

import cv2
import numpy as np
from tqdm import tqdm

np.set_printoptions(suppress=True)
np.random.seed(42)

avg_aspect_ratio = 1.75
IMG_H = 30
IMG_W = int(IMG_H / avg_aspect_ratio)

base_path = "F:/LEARN/Hackathon/Sign_detection_TestSet/dataset/Character Detection dataset/ICDAR2003"
files = glob.glob(base_path + "/cleaned_dataset_train/*/*") + glob.glob(
    base_path + "/cleaned_dataset_test/*/*"
)

### Identify frequency of each character in dataset. ###
frequency = {}
for file in files:
    label = os.path.basename(os.path.dirname(file))
    if label in frequency:
        frequency[label].append(file)
    else:
        frequency[label] = [file]
for c, f in frequency.items():
    print(c, len(f))
########################################################


### Label definition ###
label_set = string.digits + string.ascii_uppercase + string.ascii_lowercase

label_to_cls = {}
cls_to_label = {}
for i, c in enumerate(label_set):
    label_to_cls[i] = c
    cls_to_label[c] = i
#########################


### Dataset sampling and processing ###
dataset = []
for label_char, files in tqdm(frequency.items()):
    np.random.shuffle(files)

    select_count = min(70, len(files))
    files = files[:select_count]

    if label_char in cls_to_label:
        label = cls_to_label[label_char]
    else:
        continue
    for file in files:
        try:
            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_W, IMG_H))
        except:
            continue
        img = list(img.flatten())
        data = [*img, label]
        dataset.append([data])

dataset = np.array(dataset)
np.save("./dataset_v2.npy", dataset)
#######################################
