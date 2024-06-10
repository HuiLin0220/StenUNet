import cv2
import numpy as np
import copy
import os

def segments_division(binary_image):
    
    # Initialize a list to store pixel counts and arrays for each segment
    segment_pixel_counts = []
    segment_images= []
# Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a black image to draw the segments
    segmented_image = np.zeros_like(binary_image)
    
# Draw each segment and count its pixels
    for i, contour in enumerate(contours):
        color = 1  # White color for the segment
        cv2.drawContours(segmented_image, [contour], -1, color, thickness=cv2.FILLED)
        a=copy.deepcopy(segmented_image)
        segment_images.append(a)
    # Count the number of non-zero pixels in the segment
        pixel_count = cv2.countNonZero(segmented_image)
    
    
        segment_pixel_counts.append(pixel_count)
    
    # Clear the segmented image for the next iteration
        segmented_image.fill(0)
    return segment_pixel_counts,segment_images

def segments_filter_size(segment_pixel_counts, segment_images, threshold):
    out_seg = copy.deepcopy(segment_images)
    out_counts = copy.deepcopy(segment_pixel_counts)
    for j,pixel_count in enumerate (segment_pixel_counts):
        if pixel_count<threshold:
            out_counts.remove(pixel_count)
            to_remove = segment_images[j]
            
            out_seg = [arr for arr in out_seg if not np.array_equal(arr, to_remove)]
            

    return out_counts, out_seg

def binary_remove_small_segments(binary_image,threshold):
    segment_pixel_counts, segment_images=segments_division(binary_image)
    out_counts, out_seg = segments_filter_size(segment_pixel_counts,segment_images, threshold)
    out = np.zeros((512,512))
    for i in range(len(out_seg)):
        out=out+out_seg[i]
    
    return out

def remove_small_segments(input_path, output_path, threshold):
    
    for image_name in (os.listdir(input_path)):
    
        binary_image = cv2.imread(input_path + image_name, cv2.IMREAD_GRAYSCALE)
        out = binary_remove_small_segments(binary_image,threshold)
        
        cv2.imwrite(output_path + image_name, out) 
