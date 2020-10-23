### Configuration
###############################################################################
# Import Python Modules
import collections
import datetime
from google.cloud import storage
from io import BytesIO, StringIO
from operator import itemgetter
import itertools
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

# Import Project Modules
import src.config_data_processing as cdp
import src.image_manipulation as imm


### Define Functions and Classes
###############################################################################
def unnest_list_of_lists(LOL):
    """
    Unnest a list of lists
    
    Args:
        LOL (list): nested list of lists
    """
    return list(itertools.chain.from_iterable(LOL))

def index_slice_list(lst, indices):
    """
    Slice a list by a list of indices (positions)
    
    Args:
        lst (list): list to subset
        indices (list): positions to use in subsetting lst
    Returns:
        list
    """
    list_slice = itemgetter(*indices)(lst)
    if len(indices) == 1:
        return [list_slice]
    else:
        return list(list_slice)


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
    Use os to create a folder if it does not already exist on the machine
    
    Args:
        folder_path (str): folder path to conditionally create
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print_timestamp_message(f'Creating folder path {folder_path}')
    else:
        print_timestamp_message(f'Folder path {folder_path} already exists')



def read_gcs_csv_to_pandas(bucket_name, file_name, encoding = 'utf-8', header = 'infer', nrows = None, local_gcs_json_path = cdp.config_gcs_auth_json_path):
    """
    Read a csv file from a Google Cloud bucket to a local pandas.DataFrame() object
    
    Args:
        bucket_name (str): name of Google Cloud Storage bucket
        file_name (str): file name of csv object in bucket
        encoding (str): encoding of csv object. defaults to 'utf-8'
        header (str | int): header argument passed to pandas.read_csv(). defaults to 'infer'
        local_gcs_json_path (str): path on local system to Google Cloud json authentication file
    """
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = local_gcs_json_path
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    content = blob.download_as_string()
    data = StringIO(str(content, encoding))
    return pd.read_csv(data, header = header, nrows = nrows)


def write_csv_to_gcs(dframe, bucket_name, file_name, local_gcs_json_path = cdp.config_gcs_auth_json_path):
    """
    Write a csv file to a Google Cloud bucket from a local pandas.DataFrame() object
    
    Args:
        dframe (pandas.DataFrame): pandas DataFrame object to store in google cloud storage bucket
        bucket_name (str): name of Google Cloud Storage bucket
        file_name (str): file name of csv object in bucket
        local_gcs_json_path (str): path on local system to Google Cloud json authentication file
    """
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = local_gcs_json_path
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    bucket.blob(file_name).upload_from_string(dframe.to_csv(index = False), 'text/csv')
    print_timestamp_message(f'file {file_name} written to Google Cloud Storage Bucket {bucket_name}')
    

def read_gcs_numpy_array(bucket_name, file_name, local_gcs_json_path = cdp.config_gcs_auth_json_path):
    """
    Read a numpy array (.npy file) from a Google Cloud Storage Bucket
    
    Args:
        bucket_name (str): name of Google Cloud Storage bucket
        file_name (str): file name of .npy file in bucket
        local_gcs_json_path (str): path on local system to Google Cloud json authentication file
    Returns:
        numpy array
    """
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = local_gcs_json_path
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    return np.load(BytesIO(bytearray(blob.download_as_string())))


def save_np_array_to_gsc(np_array, bucket_name, file_name, local_gcs_json_path = cdp.config_gcs_auth_json_path, print_ts = True):
    """
    Save numpy array to Google Cloud Storage bucket as .npy file. Writes a temporary file to your local system, uploads to GCS, and removes from local.
    
    Args:
        np_array (numpy.array): numpy array to save in GCS
        bucket_name (str): name of Google Cloud Storage bucket
        file_name (str): file name of csv object in bucket
        local_gcs_json_path (str): path on local system to Google Cloud json authentication file
        print_ts (boolean): if True, print timestamp message. defaults to True.
    """
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = local_gcs_json_path
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    with tempfile.NamedTemporaryFile() as temp:
        temp_name = "".join([str(temp.name),".npy"])
        np.save(temp_name, np_array)
        blob = bucket.blob(file_name)
        blob.upload_from_filename(temp_name)
        os.remove(temp_name)
    print_timestamp_message(f'file {file_name} written to Google Cloud Storage Bucket {bucket_name}')
    

def list_gcs_bucket_files(bucket_name, local_gcs_json_path = cdp.config_gcs_auth_json_path):
    """
    List file names in GCS bucket
    
    Args:
        bucket_name (str): name of Google Cloud Storage bucket
        local_gcs_json_path (str): path on local system to Google Cloud json authentication file
    """
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = local_gcs_json_path
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob_files = [f for f in bucket.list_blobs()]
    file_names = [str(bf).split(',')[1].strip() for bf in blob_files]
    return file_names


def gcs_subfolder_exists(bucket_name, subfolder_name, local_gcs_json_path = cdp.config_gcs_auth_json_path):
    """
    Check whether subfolder exists in a Google Cloud Storage Bucket
    
    Args:
        bucket_name (str): name of Google Cloud Storage bucket
        subfolder_name (str): name of subfolder to check for
        local_gcs_json_path (str): path on local system to Google Cloud json authentication file
    Returns:
        boolean
    """
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = local_gcs_json_path
    file_names = list_gcs_bucket_files(bucket_name)
    subfolder_file_names = [x for x in file_names if f'/{subfolder_name}/' in x]
    return len(subfolder_file_names) > 0