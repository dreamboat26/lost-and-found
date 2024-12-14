##
# @author Meet Patel <patelmeet2012@gmail.com>
# @main.py Main file responsible for doing all OCR related work.
# @desc Created on 2022-11-19 9:22:39 pm
# @copyright APPI SASU
##

import argparse
import os
import pickle

import cv2
import numpy as np

from misc_func import (
    correct_orientation,
    detect_char_segments,
    detect_line_segments,
    detect_word_segments,
    display,
    draw_horizontal_segments,
    draw_vertical_segments,
    split_horizontally,
    split_vertically,
)
from training.classifier import ExtraTreeClassifier

os.makedirs("./results", exist_ok=True)

parser = argparse.ArgumentParser(description="OCR Module")
parser.add_argument(
    "--img_path", required=True, dest="img_path", help="Image path to infer upon."
)
parser.add_argument(
    "--model_path",
    required=True,
    dest="model_path",
    help="Model path to load the model.",
)
parser.add_argument(
    "--verbose",
    action="store_true",
    dest="verbose",
    help="Display intermediate results of image.",
)
args = parser.parse_args()

if not os.path.isfile(args.img_path):
    print("Image path is incorrect.")
if not os.path.isfile(args.model_path):
    print("Model path is incorrect.")


clf_model = pickle.load(open(args.model_path, "rb"))

img = cv2.imread(args.img_path)
if args.verbose:
    display(img)

# Correcting the orientation of the image.
img_h, img_w = img.shape[:2]
img_r, angle = correct_orientation(img)
if args.verbose:
    display(img_r)
    print(f"Image rotated at {angle:.3f} angle")

# Applying some denoising on the image for better binarization.
img_r = cv2.fastNlMeansDenoisingColored(img_r, None, 1, 3, 7, 21)

# Identifying the lines in the whole image.
vertical_segments = detect_line_segments(img_r, verbose=args.verbose)
if args.verbose:
    print("Line segments: ", vertical_segments)
line_segment_crops = split_horizontally(img_r, vertical_segments)

line_segments_plot = draw_horizontal_segments(np.copy(img_r), vertical_segments)
if args.verbose:
    display(line_segments_plot)
cv2.imwrite("./results/line_segments_plot.jpg", line_segments_plot)


predicted_string = ""
for i, crop in enumerate(line_segment_crops):
    # Identify words from each line crops.
    cv2.imwrite(f"./results/line_segments_crop_{i}.jpg", crop)

    crop_h = crop.shape[0]
    if crop_h < 10:
        # Ignore the lines with less than 10 pixel height.
        continue
    if args.verbose:
        display(crop)
    word_segments = detect_word_segments(crop, verbose=args.verbose)
    if args.verbose:
        print("Word segments: ", word_segments)
    word_segments_plot = draw_vertical_segments(np.copy(crop), word_segments)
    cv2.imwrite(f"./results/word_segments_plot_{i}.jpg", word_segments_plot)
    if args.verbose:
        display(word_segments_plot)
    word_segment_crops = split_vertically(crop, word_segments)

    for j, w_crop in enumerate(word_segment_crops):
        # Identify individual character from each of the word crops.
        cv2.imwrite(f"./results/word_segments_crop_{i}_{j}.jpg", w_crop)
        w_crop_width = w_crop.shape[1]
        if w_crop_width < 10:
            # Ignore the word crops with less than 10 pixel width.
            continue
        char_segments = detect_char_segments(w_crop, verbose=args.verbose)
        if args.verbose:
            print("Character segments: ", char_segments)
        char_segments_plot = draw_vertical_segments(np.copy(w_crop), char_segments)
        cv2.imwrite(f"./results/char_segments_plot_{i}_{j}.jpg", char_segments_plot)
        if args.verbose:
            display(char_segments_plot)
        char_segment_crops = split_vertically(w_crop, char_segments)

        for k, c_crop in enumerate(char_segment_crops):
            # Identify character from each character segment using trained model.
            cv2.imwrite(f"./results/char_segments_crop_{i}_{j}_{k}.jpg", c_crop)

            predicted_char = clf_model.predict(c_crop)
            predicted_string += predicted_char

        predicted_string += " "
    predicted_string += "\n"


print("Result:")
print(predicted_string)
