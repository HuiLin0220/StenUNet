import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.ndimage import distance_transform_edt

from .util import p


def diamond_bandpass_filter(size, angle_min, angle_max):
    # Create a grid of coordinates
    x, y = np.meshgrid(np.arange(-size, size), np.arange(-size, size))

    # Compute the angle of each coordinate
    angle = np.arctan2(y, x)
    angle = np.degrees(angle)

    # Compute the Euclidean distance of each coordinate from the origin
    distance = distance_transform_edt((x == 0) & (y == 0))

    # Create a binary mask based on the angle and distance
    p(angle_min, angle_max)
    mask = np.logical_or(np.logical_and(angle > angle_min, angle <= angle_max), np.logical_and(angle > (angle_min-180), angle <= (angle_max-180)))
    mask = np.logical_and(mask, distance <= size)

    # Create the bandpass filter
    bandpass_filter = np.zeros((2 * size, 2 * size))
    bandpass_filter[mask] = 1

    return bandpass_filter


def ddfb(image_in, plot_fft=False):
    img = image_in.copy()

    fft_img = np.fft.fftshift(np.fft.fft2(img))

    angle_sequence = [
        (0, 22.5),    # (g) 0-22.5°
        (22.5, 45),   # (h) 22.5-45°
        (45, 67.5),   # (a) 45-67.5°
        (67.5, 90),   # (b) 67.5-90°
        (90, 112.5),  # (c) 90-112.5°
        (112.5, 135), # (d) 112.5-135°
        (135, 157.5), # (e) 135-157.5°
        (157.5, 180)  # (f) 157.5-180°
    ]

    if plot_fft:
        fig, axes = plt.subplots(4, 2, figsize=(20, 40))
        axes = axes.flatten()

    dark_image_grey_fourier = fft_img.copy()

    outputs = []

    for i, angle in enumerate(angle_sequence):
        size = 256
        angle_min, angle_max = angle

        # Generate the diamond bandpass filter
        filter_ = diamond_bandpass_filter(size, angle_min, angle_max)

        out_img = dark_image_grey_fourier.copy()

        out_img_zeroes = np.zeros(out_img.shape,dtype=complex)
        out_img_zeroes[filter_ != 0] = out_img[filter_ != 0]


        if plot_fft:
            axes[i].imshow(abs(np.fft.ifft2(out_img_zeroes)),
                             cmap='gray')
            axes[i].set_title(f'angle: {angle_min}-{angle_max}°')

        outputs.append(abs(np.fft.ifft2(out_img_zeroes)))

    max_proj = np.max(outputs, axis=0)

    return max_proj


def remove_spatial_mean(img):

    fft_img = np.fft.fftshift(np.fft.fft2(img.copy()))

    result = ndimage.fourier_uniform(fft_img, size=1.5)

    return abs(np.fft.ifft2(np.fft.fftshift(result))).real