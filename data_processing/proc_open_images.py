### Overview
###############################################################################
# data source: 
#       > https://storage.googleapis.com/openimages/web/download.html
# helpful resources:
#       > https://github.com/RockyXu66/Faster_RCNN_for_Open_Images_Dataset_Keras/blob/master/Object_Detection_DataPreprocessing.ipynb
#       > https://machinelearningmastery.com/how-to-perform-object-detection-with-yolov3-in-keras/



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
import tqdm

# Import Project Modules
import src.config_data_processing as cdp
import src.image_manipulation as imm
import src.misc_functions as mf



### Define Functions
###############################################################################



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
                 img_csv_path = f'{cdp.config_train_image_csv}',
                 annotation_csv_path = f'{cdp.config_train_annotations_csv}',
                 bbox_csv_path = f'{cdp.config_train_bbox_csv}',
                 class_desc_csv_path = f'{cdp.config_class_desc_csv}',
                 image_id_col = 'ImageID',
                 image_url_col = 'OriginalURL'
                 ):
        # Initialize Arguments
        self.class_name = class_name
        self.local_gcs_json_path = local_gcs_json_path
        self.bucket_name = bucket_name
        self.img_csv_path = img_csv_path
        self.annotation_csv_path = annotation_csv_path
        self.bbox_csv_path = bbox_csv_path
        self.class_desc_csv_path = class_desc_csv_path
        self.image_id_col = image_id_col
        self.image_url_col = image_url_col
        
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
        print('this function does not do anything yet')



### Execute Functions
###############################################################################
image_retriever = OpenCVImageClassRetriever(class_name = 'Loveseat')
temp_urls, temp_bbox_df, image_ids, class_image_df = image_retriever.get_image_class_info()


image_arrays = imm.load_resize_images_from_urls(url_list = temp_urls, resize_height = cdp.config_resize_height, resize_width = cdp.config_resize_width)




# Single Image
use_image_id = image_ids[0]
use_url = class_image_df.loc[class_image_df.ImageID == use_image_id, 'OriginalURL'].values[0]
use_img_array = mf.read_url_image(use_url)
plt.imshow(use_img_array)
h, w, c = use_img_array.shape
use_bbox = temp_bbox_df.loc[temp_bbox_df.ImageID == use_image_id]
use_xmin = use_bbox['XMin'].values[0]
use_xmax = use_bbox['XMax'].values[0]
use_ymin = use_bbox['YMin'].values[0]
use_ymax = use_bbox['YMax'].values[0]






# Read Files from Storage Bucket
train_img_df = mf.read_gcs_csv_to_pandas(bucket_name = cdp.config_source_bucket_name, file_name = cdp.config_train_image_csv)
train_annot_df = mf.read_gcs_csv_to_pandas(bucket_name = cdp.config_source_bucket_name, file_name = cdp.config_train_annotations_csv)
class_desc_df = mf.read_gcs_csv_to_pandas(bucket_name = cdp.config_source_bucket_name, file_name = cdp.config_class_desc_csv, header = None)
class_bbox_df = mf.read_gcs_csv_to_pandas(bucket_name = cdp.config_source_bucket_name, file_name = cdp.config_train_bbox_csv)
class_label_dict = dict(zip(class_desc_df[1], class_desc_df[0]))
    
# Narrow Down Single Class
example_class = 'Loveseat'
example_label = class_label_dict.get(example_class)
example_class_image_ids = list(train_annot_df[(train_annot_df.LabelName == example_label) & (train_annot_df.Confidence == 1)]['ImageID'])
example_class_image_df = train_img_df[train_img_df.ImageID.isin(example_class_image_ids)]
example_class_image_urls = list(example_class_image_df['OriginalURL'])
example_class_bbox_df = class_bbox_df[class_bbox_df.ImageID.isin(example_class_image_ids)]


