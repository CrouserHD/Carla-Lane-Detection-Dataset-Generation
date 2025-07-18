import cv2
import numpy as np

def resize_image(image, factor):
    """Resizes an image by a given factor."""
    if factor == 1.0:
        return image
    new_width = int(image.shape[1] * factor)
    new_height = int(image.shape[0] * factor)
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
