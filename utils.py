##
# @author Meet Patel <patelmeet2012@gmail.com>
# @utils.py Miscellaneous utility functions file.
# @desc Created on 2022-11-15 8:02:01 pm
# @copyright APPI SASU
##

import os
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np


def center_crop(image: np.ndarray, dst_h: int) -> np.ndarray:
    """Function to center crop vertically given image.

    Args:
        image (np.ndarray): Image in [H, W, 3] dims.
        dst_h (int): Center crop height value.

    Returns:
        np.ndarray: Cropped image.
    """
    img_h = image.shape[0]
    pad_h = (img_h - dst_h) // 2
    res = image[pad_h : pad_h + dst_h, :, :]
    return res


def resize_image(
    image: np.ndarray, dst_hw: Union[Tuple, List] = (640, 640)
) -> Tuple[np.ndarray, int, int, float]:
    """Function to resize the image by padding and scalling it to provided dims.

    Args:
        image (np.ndarray): RGB Image in format [H, W, 3]
        dst_hw (Union[Tuple, List], optional): Resize dimensions. Defaults to (640, 640).

    Returns:
        Tuple[np.ndarray, int, int, float]: Tuple of resized image, padding on
            top, padding on left, scalling value to scale image to resize dims.
    """
    image_h, image_w = image.shape[:2]
    img_ar = image_h / image_w

    dst_h, dst_w = dst_hw
    if img_ar < 1.0:
        new_w = dst_w
        new_h = int(new_w * img_ar)
        pad_h = (dst_h - new_h) // 2
        pad_w = 0
    else:
        new_h = dst_h
        new_w = int(new_h / img_ar)
        pad_w = (dst_w - new_w) // 2
        pad_h = 0

    scale = image_h / new_h

    resized_image = cv2.resize(image, (new_w, new_h))
    if new_h > dst_h:
        resized_image = center_crop(resized_image, dst_h)
    else:
        resized_image = cv2.copyMakeBorder(
            resized_image,
            pad_h,
            dst_h - new_h - pad_h,
            pad_w,
            dst_w - new_w - pad_w,
            cv2.BORDER_CONSTANT,
            128,
        )

    assert resized_image.shape[:2] == tuple(dst_hw)
    return resized_image, pad_h, pad_w, scale


def plot_boxes(
    image: np.ndarray, boxes: np.ndarray, verbose=True, plot_details="min"
) -> np.ndarray:
    """Function to plot the bounding boxes on given image.

    Args:
        image (np.ndarray): Image in RGB format with [H, W, 3] dims.
        boxes (np.ndarray): Bounding boxes in [xc, yc, w, h, cls_id, cls_prob, track_id]
            format. track_id is only found in boxes if the boxes are coming from tracker.
        verbose (bool): Boolean value indicating whether to print boxes or not.
        plot_details (str): Level of details to be plotted on the image.

    Returns:
        np.ndarray: Image with boxes plotted on it.
    """
    for bb in boxes:
        xc, yc, w, h, cls_id = [int(c) for c in bb[:5]]
        if len(bb) == 7:
            track_id = bb[-1]
            if bb[-2] == -1:
                cls_prob = None
            else:
                cls_prob = bb[-2]
        else:
            track_id = None
            cls_prob = bb[-1]

        X1 = int(xc - w / 2)
        Y1 = int(yc - h / 2)
        X2 = X1 + w
        Y2 = Y1 + h
        if verbose:
            print_string = (
                f"X1={X1}, Y1={Y1}, X2={X2}, Y2={Y2}, " + f"Class-id={cls_id}"
            )
            if cls_prob is not None:
                print_string += f", Score={cls_prob*100:.3f}"
            print(print_string)

        cv2.rectangle(image, (X1, Y1), (X2, Y2), (0, 255, 0), 2)

        if track_id is None:
            if plot_details == "min":
                text_string = f"{cls_id}, {int(cls_prob*100)}"
            else:
                text_string = f"Class: {cls_id}, Prob: {int(cls_prob*100)}"
        else:
            if plot_details == "min":
                text_string = f"{int(track_id)}"
            else:
                text_string = f"Class: {cls_id}, ID: {int(track_id)}"

        (test_width, text_height), baseline = cv2.getTextSize(
            text_string, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 1
        )

        cv2.putText(
            image,
            text_string,
            (X1, Y1 - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return image


def display(image: np.ndarray) -> None:
    """Display image using cv2.

    Args:
        image (np.ndarray): Image with [H, W, 3] shape.
    """
    cv2.imshow("img", image)
    cv2.waitKey()
    cv2.destroyAllWindows()


def get_class_mapping_dict(mapping_file: str) -> Dict:
    """Function to parse the class mapping file and convert it into dict.

    Args:
        mapping_file (str): Class mapping file.

    Returns:
        Dict: Dict with mapping from class name to class id.
    """
    with open(mapping_file, "r") as file:
        lines = file.readlines()

    class_mapping = {}
    for i, line in enumerate(lines):
        line = line.replace("\n", "")
        class_mapping[line] = i

    return class_mapping


def single_image_inference(detector_model, img_path: str, output_path: str) -> None:
    """Function to run inference on single image and save results on output_path.

    Args:
        detector_model (Detector): Detection model.
        img_path (str): Image path.
        output_path (str): Output path where the result is stored.
    """
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = detector_model.detect_boxes(image)
    res_image = plot_boxes(image, boxes)
    output_file_path = os.path.join(
        output_path, os.path.splitext(os.path.basename(img_path))[0] + "_result.jpg"
    )
    res_image = cv2.cvtColor(res_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_file_path, res_image)


def convert_to_3_channel_grayscale(image: np.ndarray) -> np.ndarray:
    """Function to convert RGB image into Grayscale image with 3 channels.

    Args:
        image (np.ndarray): RGB image in [H, W, 3] format.

    Returns:
        np.ndarray: Grayscale image in [H, W, 3] format.
    """
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    stacked_gray_img = np.stack([gray_img, gray_img, gray_img], axis=2)
    return stacked_gray_img
