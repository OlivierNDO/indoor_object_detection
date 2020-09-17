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
import tempfile
from tensorflow.keras.preprocessing.image import load_img
import time
from skimage.transform import resize
import tqdm

# Import Project Modules
import src.config_data_processing as cdp
import src.misc_functions as mf

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


def extract_image_subset(img_arr, xmin, xmax, ymin, ymax):
    """
    Retrieve subset of 3d numpy array using decimal coordinates
    (i.e. portion of image with bounding box)
    Args:
        img_arr (np.array): 3d numpy array of image
        xmin (float): minimum X-coordinate (expressed as decimal)
        xmax (float): maximum X-coordinate (expressed as decimal)
        ymin (float): minimum Y-coordinate (expressed as decimal)
        ymax (float): maximum Y-coordinate (expressed as decimal)
    Returns:
        numpy.array
    """
    h, w, c = img_arr.shape
    return img_arr[int(ymin * h):int(ymax * h), int(xmin * w):int(xmax * w)]


def plot_image_bounding_box(img_arr, xmin, xmax, ymin, ymax, label,
                            box_color = 'red', text_color = 'red', 
                            fontsize = 11, linewidth = 1, y_offset = -10):
    """
    Create a matplotlib image plot with one or more bounding boxes
    Args:
        img_array (numpy.array): numpy array of image
        xmin (list): list of x-minimum coordinates (expressed as percentages)
        xmax (list): list of x-maximum coordinates (expressed as percentages)
        ymin (list): list of y-minimum coordinates (expressed as percentages)
        ymax (list): list of y-maximum coordinates (expressed as percentages)
        label (list): list of bounding box labels
        box_color (str): color to use in bounding box edge (defaults to 'red')
        text_color (str): color to use in text label (defaults to 'red')
        fontsize (int): size to use for label font (defaults to 11)
        linewidth (int): size to use for box edge line width (defaults to 1)
        y_offset (int): how far to offset text label from upper-left corner of bounding box (defaults to -10)
    """
    # Extract image dimensions and create plot object
    h, w, c = img_arr.shape
    fig,ax = plt.subplots(1)
    
    # Extract coordinates and dimensions
    for i, x in enumerate(xmin):
        
        xmin_p = int(x * w)
        xmax_p = int(xmax[i] * w)
        ymin_p = int(ymin[i] * h)
        ymax_p = int(ymax[i] * h)
        box_width = xmax_p - xmin_p
        box_height = ymax_p - ymin_p
    
        # Create rectangle and label text
        rect = patches.Rectangle((xmin_p, ymin_p), box_width, box_height, linewidth = linewidth, edgecolor = box_color, facecolor = 'none')
        ax.text(xmin_p, ymin_p + y_offset, label[i], color = text_color, fontsize = fontsize)
        ax.add_patch(rect)
    plt.imshow(img_arr)
    plt.show()


class OpenCVImageClassRetriever:
    """
    Retrieve, process, and save images from Google Open Images V6 URLs
    Args:
        class_name (str): class name corresponding to subset of images
        bucket_name (str): Google Storage Bucket to read csv files from
        img_csv_path (str): csv file path to train images information (contains URLs, image IDs, etc.)
        annotation_csv_path (str): csv file path to annotation file (object label information)
        bbox_csv_path (str): csv file path to bounding box coordinate file (bounding box dimensions)
        class_desc_csv_path (str): csv file path class description mapping
        image_id_col (str): column name convention in Google Open Images used for the image identifier
        image_url_col (str): column name convention in Google Open Images used for URL reading
    """
    
    def __init__(self, 
                 class_name,
                 local_gcs_json_path = f'{cdp.config_gcs_auth_json_path}',
                 bucket_name = f'{cdp.config_source_bucket_name}',
                 processed_bucket_subfolder = f'{cdp.config_processed_bucket_subfolder}',
                 processed_array_save_name = 'train_images.npy',
                 processed_bbox_save_name = 'train_bbox.csv',
                 processed_class_save_name = 'train_class_df.csv',
                 img_csv_path = f'{cdp.config_train_image_csv}',
                 annotation_csv_path = f'{cdp.config_train_annotations_csv}',
                 bbox_csv_path = f'{cdp.config_train_bbox_csv}',
                 class_desc_csv_path = f'{cdp.config_class_desc_csv}',
                 image_id_col = 'ImageID',
                 image_url_col = 'OriginalURL',
                 resize_height = cdp.config_resize_height,
                 resize_width = cdp.config_resize_width,
                 use_local = False,
                 local_save_path = cdp.config_temp_local_folder
                 ):
        # Initialize Arguments
        self.class_name = class_name
        self.local_gcs_json_path = local_gcs_json_path
        self.bucket_name = bucket_name
        self.processed_bucket_subfolder = processed_bucket_subfolder
        self.processed_array_save_name = processed_array_save_name
        self.processed_bbox_save_name = processed_bbox_save_name
        self.processed_class_save_name = processed_class_save_name
        self.img_csv_path = img_csv_path
        self.annotation_csv_path = annotation_csv_path
        self.bbox_csv_path = bbox_csv_path
        self.class_desc_csv_path = class_desc_csv_path
        self.image_id_col = image_id_col
        self.image_url_col = image_url_col
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.use_local = use_local
        self.local_save_path = local_save_path
        
        # Reference Google Cloud Authentication Document
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.local_gcs_json_path
        

    def get_image_class_info(self):
        # Read Csv Files
        img_df = mf.read_gcs_csv_to_pandas(bucket_name = self.bucket_name, file_name = self.img_csv_path)
        annot_df = mf.read_gcs_csv_to_pandas(bucket_name = self.bucket_name, file_name = self.annotation_csv_path)
        class_desc_df = mf.read_gcs_csv_to_pandas(bucket_name = self.bucket_name, file_name = self.class_desc_csv_path, header = None)
        class_bbox_df = mf.read_gcs_csv_to_pandas(bucket_name = self.bucket_name, file_name = self.bbox_csv_path)
        
        # Create Dictionary of Image Class Labels
        class_label_dict = dict(zip(class_desc_df[1], class_desc_df[0]))
        
        # Retrieve and return URLs, Bounding Box DataFrame, and Image IDs
        class_label = class_label_dict.get(self.class_name)
        class_image_ids = list(annot_df[(annot_df.LabelName == class_label) & (annot_df.Confidence == 1)][self.image_id_col])
        class_image_df = img_df[img_df.ImageID.isin(class_image_ids)]
        class_bbox_df = class_bbox_df[(class_bbox_df.ImageID.isin(class_image_ids)) & \
                                      (class_bbox_df.LabelName == class_label)]
        class_image_ids = list(class_bbox_df[self.image_id_col])
        class_image_urls = list(class_image_df[self.image_url_col])
        return class_image_urls, class_bbox_df, class_image_ids, class_image_df
    
    def resize_and_save_images(self):
        # Generate Class Information
        mf.print_timestamp_message(f'Getting urls, bounding boxes, and image IDs for {self.class_name} images')
        urls, bbox_df, image_ids, class_image_df = self.get_image_class_info()
        
        # Read and Resize Images
        mf.print_timestamp_message(f'Reading images from URLs and resizing to {self.resize_height} X {self.resize_width}')
        image_arrays = load_resize_images_from_urls(url_list = urls, resize_height = self.resize_height, resize_width = self.resize_width)
        image_arrays_concat = np.array(image_arrays)
        
        # Write Images to Google Cloud Storage Bucket
        image_save_name = f'{self.processed_bucket_subfolder}{self.class_name}/{self.processed_array_save_name}'
        mf.print_timestamp_message(f'Writing images to GCS bucket/folder {self.bucket_name}/{image_save_name}')
        if self.use_local:
            mf.save_np_array_to_gsc_local_path(np_array = image_arrays_concat, bucket_name = self.bucket_name, file_name = image_save_name, local_folder = self.local_save_path)
        else:
            mf.save_np_array_to_gsc(np_array = image_arrays_concat, bucket_name = self.bucket_name, file_name = image_save_name)
        
        # Write Bounding Box Csv to Google Cloud Storage Bucket
        bbox_save_name = f'{self.processed_bucket_subfolder}{self.class_name}/{self.processed_bbox_save_name}'
        mf.print_timestamp_message(f'Writing bounding box csv file to GCS bucket/folder {self.bucket_name}/{bbox_save_name}')
        mf.write_csv_to_gcs(dframe = bbox_df, bucket_name = self.bucket_name, file_name = bbox_save_name)
        
        # Write Class Info Csv to Google Cloud Storage Bucket
        class_save_name = f'{self.processed_bucket_subfolder}{self.class_name}/{self.processed_class_save_name}'
        mf.print_timestamp_message(f'Writing class image info csv file to GCS bucket/folder {self.bucket_name}/{class_save_name}')
        mf.write_csv_to_gcs(dframe = class_image_df, bucket_name = self.bucket_name, file_name = class_save_name)
        
    def get_class_image_array(self):
        # Assert Data Exists in GCS Bucket
        folder_exists = mf.gcs_subfolder_exists(bucket_name = self.bucket_name, subfolder_name = self.class_name)
        assert_msg = f"Data for image class '{self.class_name} doesn't exist. Create it with self.resize_and_save_images() method'"
        assert folder_exists, assert_msg
        
        # Read and Return Images
        image_save_name = f'{self.processed_bucket_subfolder}{self.class_name}/{self.processed_array_save_name}'
        img_array = mf.read_gcs_numpy_array(bucket_name = self.bucket_name, file_name = image_save_name)
        return img_array
    
    def get_bounding_box_df(self):
        # Assert Data Exists in GCS Bucket
        folder_exists = mf.gcs_subfolder_exists(bucket_name = self.bucket_name, subfolder_name = self.class_name)
        assert_msg = f"Data for image class '{self.class_name} doesn't exist. Create it with self.resize_and_save_images() method'"
        assert folder_exists, assert_msg
        
        # Read and Return Images
        csv_save_name = f'{self.processed_bucket_subfolder}{self.class_name}/{self.processed_bbox_save_name}'
        bbox_df = mf.read_gcs_csv_to_pandas(bucket_name = self.bucket_name, file_name = csv_save_name)
        return bbox_df
    
    def get_class_desc_df(self):
        # Assert Data Exists in GCS Bucket
        folder_exists = mf.gcs_subfolder_exists(bucket_name = self.bucket_name, subfolder_name = self.class_name)
        assert_msg = f"Data for image class '{self.class_name} doesn't exist. Create it with self.resize_and_save_images() method'"
        assert folder_exists, assert_msg
        
        # Read and Return Images
        csv_save_name = f'{self.processed_bucket_subfolder}{self.class_name}/{self.processed_class_save_name}'
        class_desc_df = mf.read_gcs_csv_to_pandas(bucket_name = self.bucket_name, file_name = csv_save_name)
        return class_desc_df

















