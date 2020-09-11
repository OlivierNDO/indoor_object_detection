### Configuration
###############################################################################
# Import Python Modules
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


### Define Functions and Classes
###############################################################################
def get_unique_count_dict(lst):
    """
    Generate and return dictionary with counts of unique items in a list
    Args:
        lst (list): list for which to generate unique element counts
    """
    key_values = collections.Counter(lst).keys()
    count_values = collections.Counter(lst).values()
    return dict(zip(key_values, count_values))


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


def load_resize_images(full_file_paths, resize_height, resize_width):
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


def print_timestamp_message(message, timestamp_format = '%Y-%m-%d %H:%M:%S'):
    """
    Print formatted timestamp followed by custom message
    Args:
        message (str): string to concatenate with timestamp
        timestamp_format (str): format for datetime string. defaults to '%Y-%m-%d %H:%M:%S'
    """
    ts_string = datetime.datetime.fromtimestamp(time.time()).strftime(timestamp_format)
    print(f'{ts_string}: {message}')


def create_folder_if_not_existing(folder_path):
    """
    Use os to create a folder if it does not already exist on the machien
    Args:
        folder_path (str): folder path to conditionally create
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print_timestamp_message(f'Creating folder path {folder_path}')
    else:
        print_timestamp_message(f'Folder path {folder_path} already exists')



def read_gcs_csv_to_pandas(bucket_name, file_name, encoding = 'utf-8', header = 'infer'):
    """
    Read a csv file from a Google Cloud bucket to a local pandas.DataFrame() object
    Args:
        bucket_name (str): name of Google Cloud Storage bucket
        file_name (str): file name of csv object in bucket
        encoding (str): encoding of csv object. defaults to 'utf-8'
        header (str | int): header argument passed to pandas.read_csv(). defaults to 'infer'
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    content = blob.download_as_string()
    data = StringIO(str(content, encoding))
    return pd.read_csv(data, header = header)