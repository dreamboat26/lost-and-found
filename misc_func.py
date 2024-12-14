from typing import Dict, List, Tuple, Union

import cv2
import imutils
import numpy as np
from scipy.signal import find_peaks


def convertToGray(img: np.ndarray) -> np.ndarray:
    """Function to convert an RGB image into Grayscale image.

    Args:
        img (np.ndarray): RGB image.

    Returns:
        np.ndarray: Grayscale image.
    """
    assert img.shape[-1] == 3, "Image should be an RGB image."
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def binarizeImage(img: np.ndarray, threshold: int = 128) -> np.ndarray:
    """Function to binarize grayscale image based on given threshold.

    Args:
        img (np.ndarray): Grayscale image.
        threshold (int, optional): Threshold for binarization. Defaults to 128.

    Returns:
        np.ndarray: Binarized image.
    """
    assert len(img.shape) == 2, "Image should be gray scale image."
    return cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


def adaptiveBinarizeImage(img: np.ndarray, block_val: int, c_val: int) -> np.ndarray:
    """Function to binarize grayscale image using adaptive threshold.

    Args:
        img (np.ndarray): Grayscale image.
        block_val (int): Block size for neighbourhood calculation of the
            adaptive binarization.
        c_val (int): Contant value used for calculation of adaptive binarization.

    Returns:
        np.ndarray: Binarized image.
    """
    assert len(img.shape) == 2, "Image should be gray scale image."
    return cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_val, C=c_val
    )


def gblur(img: np.ndarray, k_size: Tuple[int] = (3, 3)) -> np.ndarray:
    """Function to apply gaussian blur based on given kernel.

    Args:
        img (_type_): Image.
        k_size (tuple, optional): _description_. Defaults to (3, 3).

    Returns:
        np.ndarray : Blurred image.
    """
    return cv2.GaussianBlur(img, k_size, 0)


def erode(img: np.ndarray, kernel: Tuple[int] = (1, 1), iter: int = 1) -> np.ndarray:
    """Function to apply erosion on the given grayscale image.

    Args:
        img (np.ndarray): Grayscale image.
        kernel (Tuple[int], optional): Kernel size for erosion. Defaults to (1, 1).
        iter (int, optional): Number of iterations. Defaults to 1.

    Returns:
        np.ndarray: Eroded image.
    """
    # erodes away the boundaries of foreground object
    return cv2.erode(img, kernel, iterations=iter)


def dilate(img: np.ndarray, kernel: Tuple[int] = (1, 1), iter: int = 1) -> np.ndarray:
    """Function to apply dilation on the given grayscale image.

    Args:
        img (np.ndarray): Grayscale image.
        kernel (Tuple[int], optional): Kernel size for dilation. Defaults to (1, 1).
        iter (int, optional): Number of iterations. Defaults to 1.

    Returns:
        np.ndarray: Dilated image.
    """
    return cv2.dilate(
        img, kernel, iterations=iter
    )  # erodes away the boundaries of foreground object


def horizontal_projection(bg_image: np.ndarray) -> np.ndarray:
    """Function to get the horizontal projection of given grayscale image.

    Args:
        bg_image (np.ndarray): Grayscale image.

    Returns:
        np.ndarray: 1d array containing horizontal projection/histogram.
    """
    assert len(bg_image.shape) == 2, "Image should be a Grayscale image."
    hist = np.sum(bg_image, axis=1)
    return hist


def vertical_projection(bg_image: np.ndarray) -> np.ndarray:
    """Function to get the vertical projection of given grayscale image.

    Args:
        bg_image (np.ndarray): Grayscale image.

    Returns:
        np.ndarray: 1d array containing vertical projection/histogram.
    """
    assert len(bg_image.shape) == 2, "Image should be a Grayscale image."
    hist = np.sum(bg_image, axis=0)
    return hist


def correct_orientation(
    image: np.ndarray, is_rgb: bool = True
) -> Tuple[np.ndarray, float]:
    """Function to identify the correct rotation required for the image and
    rotate the image.

    Args:
        image (np.ndarray): Image.
        is_rgb (bool, optional): Boolean flag whether the provided image is RGB
            or Grayscale. Defaults to True.

    Returns:
        Tuple[np.ndarray, float]: Tuple of rotated image and rotation applied to
            the image.
    """
    if is_rgb:
        gray_img = convertToGray(image)
    else:
        gray_img = np.copy(image)
    bin_img = adaptiveBinarizeImage(gray_img, 5, 0)
    bin_img_erode = dilate(bin_img, (3, 3), 1)

    delta = 0.1
    limit = 10
    angles = np.arange(-limit, limit + delta, delta)
    scores = []

    def find_score(image, angle):
        image_r = imutils.rotate(image, angle=angle)
        hist = horizontal_projection(image_r)
        score = np.sum((hist[1:] - hist[:-1]) ** 2)
        return hist, score

    scores = []
    for angle in angles:
        hist, score = find_score(bin_img_erode, angle)
        scores.append(score)

    best_score = max(scores)
    best_angle = angles[scores.index(best_score)]

    image_corrected = imutils.rotate(image, angle=best_angle)
    return image_corrected, best_angle


def invert(bin_img: np.ndarray) -> np.ndarray:
    """Function to invert the binary image.

    Args:
        bin_img (np.ndarray): Binarized image.

    Returns:
        np.ndarray: Inverted binary image.
    """
    assert len(bin_img.shape) == 2, "Image should be a Grayscale image."
    inv_img = cv2.bitwise_not(bin_img)
    return inv_img


def visualize_horizontal_hist(hist_h: np.ndarray, img: np.ndarray) -> np.ndarray:
    """Fucntion to create a mask for visualizing the horizontal histogram /
    projection.

    Args:
        hist_h (np.ndarray): Horizontal projection/historgram of image.
        img (np.ndarray): Image.

    Returns:
        np.ndarray: Mask having the horizontal projection visualized.
    """
    img_h, img_w = img.shape[:2]
    mask = np.zeros(shape=(img_h, img_w), dtype=np.uint8)
    for i, h in enumerate(hist_h):
        new_h = int(h / 255)
        mask[i, :new_h] = 255

    return mask


def visualize_vertical_hist(hist_v: np.ndarray, img: np.ndarray) -> np.ndarray:
    """Fucntion to create a mask for visualizing the vertical histogram /
    projection.

    Args:
        hist_h (np.ndarray): Vertical projection/historgram of image.
        img (np.ndarray): Image.

    Returns:
        np.ndarray: Mask having the vertical projection visualized.
    """
    img_h, img_w = img.shape[:2]
    mask = np.zeros(shape=(img_h, img_w), dtype=np.uint8)
    for i, v in enumerate(hist_v):
        new_v = int(v / 255)
        mask[(img_h - new_v) :, i] = 255

    return mask


def detect_line_segments(
    img_rgb: np.ndarray, verbose: bool = False
) -> Tuple[Tuple[int]]:
    """Function to detect the separating lines to identify each line segment in
    the image.

    Args:
        img_rgb (np.ndarray): RGB Image
        verbose (bool, optional): Boolean flag to display intermediate results.
            Defaults to False.

    Returns:
        Tuple[Tuple[int]]: Each tuple contains start and end point value for each of the
            segments.
    """
    img_h, img_w = img_rgb.shape[:2]
    gray_img = convertToGray(img_rgb)
    bin_img = adaptiveBinarizeImage(gray_img, 5, 0)

    if verbose:
        display(bin_img)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_DILATE, kernel)
    if verbose:
        display(bin_img)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bin_img_erode = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)
    if verbose:
        display(bin_img_erode)
    hist_h = horizontal_projection(bin_img_erode)
    if verbose:
        mask_h = visualize_horizontal_hist(hist_h, bin_img_erode)
        display(mask_h)

    # Dividing img by 4 with an assumption that the 3 separating lines are
    # required to segment image with 2 lines.
    # peaks_h, _ = find_peaks(hist_h, distance=img_h / 4)
    # peaks_h = list(peaks_h)

    peaks_h = find_peak_segments(hist_h, img_h / 5)
    return peaks_h


def draw_horizontal_segments(
    img: np.ndarray, segment_groups: Tuple[Tuple[int]]
) -> np.ndarray:
    """Function to use the separating lines and draw it on the provided image.

    Args:
        img (np.ndarray): RGB Image.
        segment_groups (Tuple[Tuple[int]]): Separating groups to differentiate each
            line segment.

    Returns:
        np.ndarray: Image with separating lines plotted on it.
    """
    img_w = img.shape[1]
    for start, end in segment_groups:
        cv2.line(img, (0, start), (img_w, start), (0, 0, 255), 1, cv2.LINE_AA)
        cv2.line(img, (0, end), (img_w, end), (0, 0, 255), 1, cv2.LINE_AA)
    return img


def split_horizontally(
    img: np.ndarray, segment_groups: Tuple[Tuple[int]]
) -> List[np.ndarray]:
    """Function to split the given image horizontally based on line segments.

    Args:
        img (np.ndarray): RGB Image
        segment_groups (Tuple[Tuple[int]]): Separating groups for splitting the image.

    Returns:
        List[np.ndarray]: List of image crops.
    """
    result = []
    for start, end in segment_groups:
        crop = img[start:end, :, :]
        result.append(crop)
    return result


def find_peak_segments(
    hist: np.ndarray, min_dist: float, thresh: float = 0.125
) -> Tuple[Tuple[int]]:
    """Function to identify the peak values in the given histogram / projection.

    Args:
        hist (np.ndarray): Histogram / Projection of the binary image.
        min_dist (float): Min distance of start and end point of each segment
            for filtering purpose.
        thresh (float, optional): Threshold value to filter the histogram values.
            Defaults to 0.125.

    Returns:
        Tuple[Tuple[int]]: Each tuple contains start and end value of Separating
            segments.
    """
    hist = (hist - np.min(hist)) / (np.max(hist) - np.min(hist))
    non_zero_peaks = np.where(hist > thresh)[0]
    zero_peaks = np.where(hist <= thresh)[0]

    start = non_zero_peaks[0]

    separating_points = []
    while True:
        end = zero_peaks[zero_peaks > start]
        if len(end) == 0:
            break

        end = end[0]

        if (end - start) > min_dist:
            separating_points.append(start)
            separating_points.append(end)

        start = non_zero_peaks[non_zero_peaks > end]
        if len(start) == 0:
            break
        start = start[0]

    separating_groups = [
        (a, b) for a, b in zip(separating_points[0::2], separating_points[1::2])
    ]
    return separating_groups


# unused function
def word_separating_line(hist: np.ndarray, img_h: int) -> List[float]:
    """Function to identify the word separating lines from histogram / projection.

    Args:
        hist (np.ndarray): Vertical Histogram / projection of the image.
        img_h (int): Image height.

    Returns:
        List[float]: List of values which separates words in the image.
    """
    hist = hist / 255

    start_idx = -1
    end_idx = -1
    segments = []
    for i in range(len(hist)):
        if hist[i] >= img_h * 0.95:
            if start_idx == -1:
                start_idx = i
            else:
                continue
        else:
            if start_idx != -1:
                end_idx = i
                mid_point = int((start_idx + end_idx) / 2)
                segments.append(mid_point)
                start_idx = -1
                end_idx = -1
            else:
                continue

    return segments


def detect_word_segments(
    img_rgb: np.ndarray, verbose: bool = False
) -> Tuple[Tuple[int]]:
    """Function to detect the separating lines to identify each word in the
    given image.

    Args:
        img_rgb (np.ndarray): RGB Image
        verbose (bool, optional): Boolean flag to display intermediate results.
            Defaults to False.

    Returns:
        Tuple[Tuple[int]]: Each tuple contains start and end point value for
            each of the word segments.
    """
    img_h, img_w = img_rgb.shape[:2]
    gray_img = convertToGray(img_rgb)
    bin_img = adaptiveBinarizeImage(gray_img, 5, 0)

    if verbose:
        display(bin_img)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_DILATE, kernel)

    hist_v = vertical_projection(bin_img)
    if verbose:
        mask_v = visualize_vertical_hist(hist_v, bin_img)
        display(mask_v)

    # # Dividing img by 4 with an assumption that the 3 separating lines are
    # # required to segment the given line image crop with 2 words in it.
    # peaks_v, _ = find_peaks(inv_hist_v, distance=img_w / 4)
    # peaks_v = list(peaks_v)

    peaks_v = find_peak_segments(hist_v, img_w / 15, thresh=0.3)
    return peaks_v


def split_vertically(
    img: np.ndarray, segment_groups: Tuple[Tuple[int]]
) -> List[np.ndarray]:
    """Function to split the given image vertically based on given segments.

    Args:
        img (np.ndarray): RGB Image
        segment_groups (Tuple[Tuple[int]]): Separating groups for splitting the image.

    Returns:
        List[np.ndarray]: List of image crops.
    """
    result = []
    for start, end in segment_groups:
        crop = img[:, start:end, :]
        result.append(crop)
    return result


def draw_vertical_segments(
    img: np.ndarray, segment_groups: Tuple[Tuple[int]]
) -> np.ndarray:
    """Function to use the separating lines and draw it on the provided image.

    Args:
        img (np.ndarray): RGB Image.
        segment_groups (Tuple[Tuple[int]]): Separating groups to differentiate
            each word/character segment.

    Returns:
        np.ndarray: Image with separating lines plotted on it.
    """
    img_h = img.shape[0]
    for start, end in segment_groups:
        cv2.line(img, (start, 0), (start, img_h), (0, 0, 255), 1, cv2.LINE_AA)
        cv2.line(img, (end, 0), (end, img_h), (0, 0, 255), 1, cv2.LINE_AA)
    return img


def detect_char_segments(
    img_rgb: np.ndarray, verbose: bool = False
) -> Tuple[Tuple[int]]:
    """Function to detect the separating lines to identify each character
    segment in the image.

    Args:
        img_rgb (np.ndarray): RGB Image
        verbose (bool, optional): Boolean flag to display intermediate results.
            Defaults to False.

    Returns:
        Tuple[Tuple[int]]: Each tuple contains start and end point value for
            each of the character segments.
    """
    img_h, img_w = img_rgb.shape[:2]
    gray_img = convertToGray(img_rgb)
    bin_img = adaptiveBinarizeImage(gray_img, 11, 0)
    if verbose:
        display(bin_img)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_DILATE, kernel, iterations=1)
    if verbose:
        display(bin_img)

    hist_v = vertical_projection(bin_img)
    if verbose:
        mask_v = visualize_vertical_hist(hist_v, bin_img)
        display(mask_v)

    # # Dividing img by 4 with an assumption that the 3 separating lines are
    # # required to segment the given line image crop with 2 words in it.
    # peaks_v, _ = find_peaks(inv_hist_v) #, distance=img_w / 5)
    # peaks_v = list(peaks_v)

    peaks_v = find_peak_segments(hist_v, img_w / 30, 0.5)
    return peaks_v


def display(img: np.ndarray, window_name="img") -> None:
    """Function to display an image and wait till user provides any input.

    Args:
        img (np.ndarray): Image to be shown.
        window_name (str, optional): Window name. Defaults to 'img'.
    """
    cv2.imshow(window_name, img)
    cv2.waitKey()
    cv2.destroyAllWindows()
