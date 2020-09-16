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
import tempfile
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


def write_csv_to_gcs(dframe, bucket_name, file_name):
    """
    Write a csv file to a Google Cloud bucket from a local pandas.DataFrame() object
    Args:
        dframe (pandas.DataFrame): pandas DataFrame object to store in google cloud storage bucket
        bucket_name (str): name of Google Cloud Storage bucket
        file_name (str): file name of csv object in bucket
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    bucket.blob(file_name).upload_from_string(dframe.to_csv(index = False), 'text/csv')
    print_timestamp_message(f'file {file_name} written to Google Cloud Storage Bucket {bucket_name}')
    

def read_gcs_numpy_array(bucket_name, file_name):
    """
    Read a numpy array (.npy file) from a Google Cloud Storage Bucket
    Args:
        bucket_name (str): name of Google Cloud Storage bucket
        file_name (str): file name of .npy file in bucket
    Returns:
        numpy array
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    return np.load(BytesIO(bytearray(blob.download_as_string())))


def save_np_array_to_gsc(np_array, bucket_name, file_name):
    """
    Save numpy array to Google Cloud Storage bucket as .npy file.
    Writes a temporary file to your local system, uploads to GCS, and removes from local.
    Args:
        np_array (numpy.array): numpy array to save in GCS
        bucket_name (str): name of Google Cloud Storage bucket
        file_name (str): file name of csv object in bucket
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    with tempfile.NamedTemporaryFile() as temp:
        temp_name = "".join([str(temp.name),".npy"])
        np.save(temp_name, np_array)
        blob = bucket.blob(file_name)
        blob.upload_from_filename(temp_name)
        os.remove(temp_name)
    print_timestamp_message(f'file {file_name} written to Google Cloud Storage Bucket {bucket_name}')
        
