import cv2
import numpy as np
from scipy.ndimage import white_tophat, black_tophat
from skimage.morphology import disk


def WTH(img, B):
    return cv2.morphologyEx(img.copy(), cv2.MORPH_TOPHAT, B)


def BTH(img, B):
    return cv2.morphologyEx(img.copy(), cv2.MORPH_BLACKHAT, B)


def WTH_scipy(img, B):
    return white_tophat(input=img.copy(), structure=B)


def BTH_scipy(img, B):
    return black_tophat(input=img.copy(), structure=B)


def adjust_scale(image):
    # Find the minimum and maximum values in the array
    min_val = np.min(image)
    max_val = np.max(image)

    # Scale the array to the range 0 to 255
    adjusted_image = ((image - min_val) / (max_val - min_val)) * 255

    # Convert the values to integers
    adjusted_image

    return adjusted_image

def multi_scale(original, k=3, n=19, i=2):
    img = original.copy().astype(np.uint8)
    scales = [disk(_ // 2) for _ in range(k, n, i)]

    wth_n = np.asarray([WTH(img.copy(), scale) for scale in scales])
    bth_n = np.asarray([BTH(img.copy(), scale) for scale in scales])

    wth_neighbor = np.diff(wth_n, axis=0)
    bth_neighbor = np.diff(bth_n, axis=0)

    f_c_w = np.max(wth_n, axis=0)
    f_c_b = np.max(bth_n, axis=0)

    f_d_w = np.max(wth_neighbor, axis=0)
    f_d_b = np.max(bth_neighbor, axis=0)

    white_ = (f_c_w + f_d_w)
    black_ = (f_c_b + f_d_b)
    f_en = img + adjust_scale(white_) - adjust_scale(black_)

    return adjust_scale(f_en)