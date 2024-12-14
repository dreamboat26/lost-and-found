##
# @author Meet Patel <patelmeet2012@gmail.com>
# @main.py Helper code for analysing and cleaning the ICDAR 2003 dataset.
# @desc Created on 2022-11-20 8:23:11 pm
# @copyright APPI SASU
##

import glob
import os
import shutil
import xml.etree.ElementTree as ET

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

output_dir = "./cleaned_dataset_test/"
os.makedirs(output_dir, exist_ok=False)

xml_path = "./test_set/char.xml"
tree = ET.parse(xml_path)

root = tree.getroot()

avg_h, avg_w, avg_ar = [], [], []

special_char = {'"': "quotation_mark", "?": "ques_mark", ":": "colon"}

for img_data in tqdm(root.findall("image")):
    img_path = os.path.join("./test_set/", img_data.get("file"))
    label = img_data.get("tag")
    if label in special_char:
        label = special_char[label]

    img = cv2.imread(img_path)
    img_h, img_w = img.shape[:2]

    avg_h.append(img_h)
    avg_w.append(img_w)
    avg_ar.append(img_h / img_w)

    label_dir = os.path.join(output_dir, label)
    os.makedirs(label_dir, exist_ok=True)

    total_files = len(glob.glob(label_dir + "/*.jpg"))
    output_path = os.path.join(label_dir, os.path.basename(img_path))
    if os.path.isfile(output_path):
        output_path = os.path.join(
            label_dir,
            os.path.splitext(os.path.basename(img_path))[0]
            + f"_{np.random.randint(1000)}"
            + os.path.splitext(os.path.basename(img_path))[1],
        )
    shutil.copy(img_path, output_path)


num_bins = len(set(avg_ar))
print("Average aspect ratio:", np.mean(avg_ar))

n, bins, patches = plt.hist(avg_ar, num_bins, density=1, color="green", alpha=0.7)

plt.xlabel("X-Axis")
plt.ylabel("Y-Axis")

plt.title("Aspect ratio distribution", fontweight="bold")

plt.show()
