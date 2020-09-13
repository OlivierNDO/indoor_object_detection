### Configuration
###############################################################################

# Import Modules
import collections
import datetime
from google.cloud import storage
from io import BytesIO, StringIO
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
import numpy as np
import os
import pandas as pd
from PIL import Image
import requests
from tensorflow.keras.preprocessing.image import load_img
import time
from skimage.transform import resize
import tqdm


### Define Functions
###############################################################################


def img_add_flip(arr, flip_horiz = True, flip_vert = False):
    """
    Flip numpy array horizontally and/or vertically
    Args:
        arr: three dimensional numpy array
        flip_horiz: flip image horizontally
        flip_vert: flip image vertically
    """
    assert len(arr.shape) == 3, "'arr' input array must be three dimensional"
    arr_copy = arr.copy()
    if flip_horiz:
        arr_copy = np.fliplr(arr_copy)
    if flip_vert:
        arr_copy = np.flipud(arr_copy)
    return arr_copy



def read_url_image(url):
    """
    Read image from URL with .jpg or .png extension
    Args:
        url (str): url character string
    Returns:
        numpy array
    """
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return np.array(img)

 

def load_resize_images_from_files(full_file_paths, resize_height, resize_width):
    """
    Load images and resize according to function arguments
    Args:
        full_file_paths: list of saved image files
        resize_height: height of resized output images
        resize_width: width of resized output images
    Depdendencies:
        numpy
        skimage.transform.resize
        tensorflow.keras.preprocessing.image.load_img
    Returns:
        numpy array of resized images
    """
    read_images = [load_img(c) for c in full_file_paths]
    resized_images = [resize(np.array(ri), (resize_height, resize_width)) for ri in read_images]
    return np.array(resized_images)




def load_resize_images_from_urls(url_list, resize_height, resize_width):
    """
    Load images from list of URLs and resize according to function arguments
    Args:
        url_list: list of image URLs
        resize_height: height of resized output images
        resize_width: width of resized output images
    Depdendencies:
        numpy
        skimage.transform.resize
        tensorflow.keras.preprocessing.image.load_img
    Returns:
        4d numpy array of resized images
    """
    read_images = []
    for i, x in tqdm.tqdm(enumerate(url_list)):
        try:
            img = read_url_image(x)
            resized_img = resize(img[:,:,:3], (resize_height, resize_width))
            read_images.append(resized_img)
        except:
            read_images.append(np.empty((resize_width, resize_height, 3)))
    return read_images

