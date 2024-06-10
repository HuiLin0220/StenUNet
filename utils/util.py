#Syntax Project Utils
#Tom Liu

"""Utility functions for videos, plotting and computing performance metrics."""

import functools
import json
import os

import cv2  # pytype: disable=attribute-error
import matplotlib
import numpy as np
import torch
import tqdm
from PIL import Image
from matplotlib import pyplot as plt
from shapely.geometry import Polygon
from skimage import measure
# import precision_recall_curve
from sklearn.metrics import precision_recall_curve, auc, roc_curve

global verbose
verbose = True

def image_to_array(image_path):
    image = Image.open(image_path)
    array = np.array(image)
    return array

def toggle_verbose(verbosity=None):
    global verbose
    verbose = not verbose if not verbosity else verbosity

def verbose_mode(func, verbose_override=None):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        global verbose
        if verbose_override or verbose is True:
            res = func(*args, **kwargs)
        else:
            res = None
        return res

    return wrapper

@verbose_mode
def p(*args, func=print):
    return func(*args)

def loadvideo(filename: str) -> np.ndarray:
    """Loads a video from a file.

    Args:
        filename (str): filename of video

    Returns:
        A np.ndarray with dimensions (channels=3, frames, height, width). The
        values will be uint8's ranging from 0 to 255.

    Raises:
        FileNotFoundError: Could not find `filename`
        ValueError: An error occurred while reading the video
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    capture = cv2.VideoCapture(filename)

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # empty numpy array of appropriate length, fill in when possible from front
    v = np.zeros((frame_count, frame_height, frame_width, 3), np.uint8)

    for count in range(frame_count):
        ret, frame = capture.read()
        if not ret:
            raise ValueError("Failed to load frame #{} of {}.".format(count, filename))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        v[count, :, :] = frame

    v = v.transpose((3, 0, 1, 2))

    return v


def savevideo(filename: str, array: np.ndarray, fps:int):
    """Saves a video to a file.

    Args:
        filename (str): filename of video
        array (np.ndarray): video of uint8's with shape (channels=3, frames, height, width)
        fps (float or int): frames per second

    Returns:
        None
    """

    c, _, height, width = array.shape

    if c != 3:
        raise ValueError("savevideo expects array of shape (channels=3, frames, height, width), got shape ({})".format(", ".join(map(str, array.shape))))
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    for frame in array.transpose((1, 2, 3, 0)):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)


def get_mean_and_std(dataset: torch.utils.data.Dataset,
                     samples: int = 128,
                     batch_size: int = 8,
                     num_workers: int = 4):
    """Computes mean and std from samples from a Pytorch dataset.

    Args:
        dataset (torch.utils.data.Dataset): A Pytorch dataset.
            ``dataset[i][0]'' is expected to be the i-th video in the dataset, which
            should be a ``torch.Tensor'' of dimensions (channels=3, frames, height, width)
        samples (int or None, optional): Number of samples to take from dataset. If ``None'', mean and
            standard deviation are computed over all elements.
            Defaults to 128.
        batch_size (int, optional): how many samples per batch to load
            Defaults to 8.
        num_workers (int, optional): how many subprocesses to use for data
            loading. If 0, the data will be loaded in the main process.
            Defaults to 4.

    Returns:
       A tuple of the mean and standard deviation. Both are represented as np.array's of dimension (channels,).
    """

    if samples is not None and len(dataset) > samples:
        indices = np.random.choice(len(dataset), samples, replace=False)
        dataset = torch.utils.data.Subset(dataset, indices)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    n = 0  # number of elements taken (should be equal to samples by end of for loop)
    s1 = 0.  # sum of elements along channels (ends up as np.array of dimension (channels,))
    s2 = 0.  # sum of squares of elements along channels (ends up as np.array of dimension (channels,))
    for (x, *_) in tqdm.tqdm(dataloader):
        x = x.transpose(0, 1).contiguous().view(3, -1)
        n += x.shape[1]
        s1 += torch.sum(x, dim=1).numpy()
        s2 += torch.sum(x ** 2, dim=1).numpy()
    mean = s1 / n  # type: np.ndarray
    std = np.sqrt(s2 / n - mean ** 2)  # type: np.ndarray

    mean = mean.astype(np.float32)
    std = std.astype(np.float32)

    return mean, std


def bootstrap(a, b, func, samples=10000):
    """Computes a bootstrapped confidence intervals for ``func(a, b)''.

    Args:
        a (array_like): first argument to `func`.
        b (array_like): second argument to `func`.
        func (callable): Function to compute confidence intervals for.
            ``dataset[i][0]'' is expected to be the i-th video in the dataset, which
            should be a ``torch.Tensor'' of dimensions (channels=3, frames, height, width)
        samples (int, optional): Number of samples to compute.
            Defaults to 10000.

    Returns:
       A tuple of (`func(a, b)`, estimated 5-th percentile, estimated 95-th percentile).
    """
    a = np.array(a)
    b = np.array(b)

    bootstraps = []
    for _ in range(samples):
        ind = np.random.choice(len(a), len(a))
        bootstraps.append(func(a[ind], b[ind]))
    bootstraps = sorted(bootstraps)

    return func(a, b), bootstraps[round(0.05 * len(bootstraps))], bootstraps[round(0.95 * len(bootstraps))]


# Based on https://nipunbatra.github.io/blog/2014/latexify.html
def latexify():
    """Sets matplotlib params to appear more like LaTeX.

    Based on https://nipunbatra.github.io/blog/2014/latexify.html
    """
    params = {'backend': 'pdf',
              'axes.titlesize': 8,
              'axes.labelsize': 8,
              'font.size': 8,
              'legend.fontsize': 8,
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,
              'font.family': 'DejaVu Serif',
              'font.serif': 'Computer Modern',
              }
    matplotlib.rcParams.update(params)


def dice_similarity_coefficient(inter, union):
    """Computes the dice similarity coefficient.

    Args:
        inter (iterable): iterable of the intersections
        union (iterable): iterable of the unions
    """
    return 2 * sum(inter) / (sum(union) + sum(inter))


def is_cropped(image, single_=True, thresh=0.7):
    '''
    determine if angiogram dicom has cropped image. If it is cropped will return False

    Options include doing a random image or sampling all clips and taking the consensus agreement
    '''

    def _is_cropped(img):

        x = img > 50  # img.mean()
        z = np.where(x)
        temp_img = img[z[0][0]:z[0][-1], z[1][0]:z[1][-1]]

        w1, h1 = img.shape
        w2, h2 = temp_img.shape

        # If the image is cropped then return True, we will want to exclude
        if w2 / w1 < thresh or h2 / h1 < thresh:
            return True
        else:
            return False

    frames = image.shape[0]

    cropped_ = []

    if single_:
        i = np.random.randint(frames)
        return _is_cropped(image[i])
    else:
        for i in range(frames):
            is_cropped_ = _is_cropped(image[i])
            cropped_.append(is_cropped_)

    if sum(cropped_) > frames / 2:
        return True
    else:
        return False


import albumentations as A
def is_cropped_dice_method(img, threshold=0.6):
    '''
    Use dice method and adaptive threshold to determine if the angiogram image (f, x, w) is cropped

    Returns: bool, dice_ score, bounding box coordinates

    '''
    x = img.copy()

    if len(x.shape) == 3:
    # Reduce exposure if general cropping detected
        top_row = x[:, 0, :].mean()
        left_column = x[:, :, 0].mean()
        if top_row < 40 or left_column < 40:
            x = np.where(x > 30, x - 30, 0)
    else:
        top_row = x[0, :].mean()
        left_column = x[:, 0].mean()
        if top_row < 40 or left_column < 40:
            x = np.where(x > 30, x - 30, 0)

    # Creating the kernel(2d convolution matrix)
    # This is a sharpening kernel
    kernel_ = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]])

    # Applying the filter2D() function
    x = cv2.filter2D(src=x, ddepth=-1, kernel=kernel_)

    # Use median blur to smooth lines
    x = cv2.medianBlur(x, 5)

    #Set hard threshold
    thresh = 120 if x.mean() > 70 else x.mean()
    x_mask = ((x > thresh) * x).mean(axis=0)

    x_out = np.where(x_mask <= 0, 0, 1)

    # Get the mean intensity value
    x_mean = x.copy()

    # Obtain a mask that thresholds based on the average intensity value of the image
    threshold_img = cv2.adaptiveThreshold(x_mean.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 499, 5)

    #Calculate centroid of adaptive thresholded image
    contours, _ = cv2.findContours(threshold_img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    x1, y1, w1, h1 = cv2.boundingRect(c)

    p1 = np.array([x1 + w1 // 2, y1 + h1 // 2])

    #Calculate centroid of thresholded image
    contours, _ = cv2.findContours(x_out.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    x2, y2, w2, h2 = cv2.boundingRect(c)

    p2 = np.array([x2 + w2 // 2, y2 + h2 // 2])

    #Determine centroid distance from true center
    center = np.array([256, 256])
    a = np.linalg.norm(center - p1)
    b = np.linalg.norm(center - p2)

    #examine whether adaptive or regular threshold can be used
    if a < b:
        x, y, w, h = x1, y1, w1, h1
    else:
        x, y, w, h = x2, y2, w2, h2

    new_x = cv2.adaptiveThreshold(x_mean.astype(np.uint8), 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 499,5)

    # calculate similarity coefficient
    dice_ = dice_score(np.logical_not(new_x), np.ones([512, 512])).mean()
    # dice_ = dice_score(x_out, np.ones([512, 512])).mean()

    if dice_ > threshold:
        return False, dice_, [x, y, w, h]
    else:
        return True, dice_, [x, y, w, h]

def resize_image(img_arr: np.ndarray, bboxes=[], size_=(224, 224)):
    """
    :param img_arr: original image as a numpy array
    :param bboxes: bboxes as numpy array where each row is 'x_min', 'y_min', 'x_max', 'y_max', "class_id"
    :param h: resized height dimension of image
    :param w: resized weight dimension of image
    :return: dictionary containing {image:transformed, bboxes:['x_min', 'y_min', 'x_max', 'y_max', "class_id"]}

    PASCAL VOC: xtl, ytl, xbr, ybr
    """
    # create resize transform pipeline
    h, w = size_
    # Declare an augmentation pipeline

    if not bboxes or (not bboxes[0] and isinstance(bboxes[0], list)):
        transform = A.Compose([
            A.Resize(height=h, width=w, always_apply=True)]
        )

        transformed = transform(image=img_arr)

    else:
        transform = A.Compose([
            A.Resize(height=h, width=w, always_apply=True)],
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])
        )

        transformed = transform(image=img_arr, bboxes=[bboxes], labels=[0])

    return transformed['image'], transformed['bboxes'][0]

def crop_image(img_arr:np.ndarray, crop_coords, bboxes_):

    """
    :param img_arr: original image as a numpy array
    :param bboxes: bboxes as numpy array where each row is 'x_min', 'y_min', 'x_max', 'y_max', "class_id"
    :return: dictionary containing {image:transformed, bboxes:['x_min', 'y_min', 'x_max', 'y_max', "class_id"]}

    PASCAL VOC: xtl, ytl, xbr, ybr
    """
    x1, y1, x2, y2 = crop_coords

    # create resize transform pipeline
    h, w = img_arr.shape[:2]
    # Declare an augmentation pipeline
    transform = A.Compose([
        A.Crop(x_min=x1, y_min=y1, x_max=x2, y_max=y2, p=1.0),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    # apply transform
    transformed = transform(image=img_arr, bboxes=[bboxes_], class_labels=[0])

    # return transformed image and bboxes
    return transformed['image'], transformed['bboxes'][0]

def auto_crop(image, image_to_apply_to=None, tighten_pixels=10):
    cropped_, dice_score_, [x, y, w, h] = is_cropped_dice_method(image.astype(np.uint8))

    if cropped_:
        #tighten crop box
        x = x + tighten_pixels
        y = y + tighten_pixels
        w = w - (tighten_pixels * 2)
        h = h - (tighten_pixels * 2)

    if image_to_apply_to:
        image = image_to_apply_to.copy()
    output = crop_image(image.astype(np.uint8), crop_coords=[x, y, x+w, y+h], bboxes_=[x, y, x+w, y+h])
    output = resize_image(output[0], bboxes=output[1], size_=(512, 512))

    return output[0]

def create_targetlist(input_df, path_to_videos, out_path, path_to_crosswalk='', split=False):
    '''
    Args:
        input_df:
        path_to_videos:
        out_path:
        path_to_crosswalk:
        split:

    Returns: None
    Creates a tabular label for comparing predictions
    '''
    pass

def dice_score(x, y):
    """Computes the dice similarity coefficient.

    Args:
        inter (iterable): iterable of the intersections
        union (iterable): iterable of the unions
    """

    inter = x * y

    return (2 * np.sum(inter)) / (np.sum(x) + np.sum(y))

# __all__ = ["video", "segmentation", "loadvideo", "savevideo", "get_mean_and_std", "bootstrap", "latexify", "dice_similarity_coefficient"]

def pr_curve(y_true, y_pred, destination, plot=True):
    """Computes the precision-recall curve.

    Args:
        y_true (iterable): iterable of the ground truth
        y_pred (iterable): iterable of the predictions
        plot (bool): whether to plot the curve
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    if plot:
        plt.plot(recall, precision)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve (AUC={auc(recall, precision):.2f})")

    #save plot at desintation
    plt.savefig(destination)

def roc_curve_(y_true, y_pred, destination, plot=True):
    """Computes the ROC curve.

    Args:
        y_true (iterable): iterable of the ground truth
        y_pred (iterable): iterable of the predictions
        plot (bool): whether to plot the curve
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    if plot:
        plt.plot(fpr, tpr)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve (AUC={auc(fpr, tpr):.2f})")

    #save plot at desintation
    plt.savefig(destination)

def image_to_array(image_path):
    image = Image.open(image_path)
    array = np.array(image)
    return array

def rgb_to_grayscale(image_path):
    image = Image.open(image_path)
    grayscale_image = image.convert('L')
    array = np.array(grayscale_image)
    return array


def generate_resut_json(empty_json_path, result_json_path, result_mask, mode):
    
    with open(empty_json_path) as file:
        gt = json.load(file)
    empty_submit = dict()
    empty_submit["images"] = gt["images"]
    empty_submit["categories"] = gt["categories"]
    empty_submit["annotations"] = []
    
    gt_mask = result_mask
    
    count_anns = 1
    for img_id, img in enumerate(gt_mask, 0):
        for cls_id, cls in enumerate(img, 0):
            contours = measure.find_contours(cls)
            for contour in contours:            
                for i in range(len(contour)):
                    row, col = contour[i]
                    contour[i] = (col - 1, row - 1)

                # Simplify polygon
                poly = Polygon(contour)
                poly = poly.simplify(1.0, preserve_topology=False)
            
                if(poly.is_empty):
                    continue
                segmentation = np.array(poly.exterior.coords).ravel().tolist()
                new_ann = dict()
                new_ann["id"] = count_anns
                new_ann["image_id"] = img_id+1
                if mode=='seg':
                  new_ann["category_id"] = cls_id+1
                else:
                  new_ann["category_id"] = 26
                new_ann["segmentation"] = [segmentation]
                new_ann["area"] = poly.area
                x, y = contour.min(axis=0)
                w, h = contour.max(axis=0) - contour.min(axis=0)
                new_ann["bbox"]  = [int(x), int(y), int(w), int(h)]
                new_ann["iscrowd"] = 0
                new_ann["attributes"] = {
                "occluded": False
                }
                count_anns += 1
                empty_submit["annotations"].append(new_ann.copy())
   
    with open(result_json_path, "w") as file:
        json.dump(empty_submit, file)

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
