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

# Import Project Modules
import src.config_data_processing as cdp
import src.image_manipulation as imm
import src.misc_functions as mf




### Define Functions
###############################################################################

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
        return class_image_urls, class_bbox_df, class_image_ids
    
    
    def resize_and_save_images(self):
        print('this function does not do anything yet')



### Execute Functions
###############################################################################
image_retriever = OpenCVImageClassRetriever(class_name = 'Loveseat')
temp_urls, temp_bbox_df, image_ids = image_retriever.get_image_class_info()






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











    
    

def plot_image_bounding_box(img_array, xmin, xmax, ymin, ymax):
    """
    Create a matplotlib image plot with one or more bounding boxes
    Args:
        img_array (numpy.array): numpy array of image
        xmin (list): list of x-minimum coordinates (expressed as percentages)
        xmax (list): list of x-maximum coordinates (expressed as percentages)
        ymin (list): list of y-minimum coordinates (expressed as percentages)
        ymax (list): list of y-maximum coordinates (expressed as percentages)
    """
    # Extract image dimensions and create plot object
    h, w, c = img.shape
    fig,ax = plt.subplots(1)
    ax.imshow(img)
    
    for i, xM in enumerate(xmin):
        # Extract coordinates and dimensions
        xmin_p = int(xM * w)
        xmax_p = int(xmax[i] * w)
        ymin_p = int(ymin[i] * h)
        ymax_p = int(ymax[i] * h)
        coords = [xmin_p, xmax_p, ymin_p, ymax_p]
        print(coords)
        upper_left_point = (xmin_p, ymax_p)
        box_width = xmax_p - xmin_p
        box_height = ymax_p - ymin_p
        print(upper_left_point)
        
        # Create rectangle
        rect = patches.Rectangle((xmin_p, ymax_p), box_width, box_height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()





i = 32
img_id = image_ids[i]
img_bbox = temp_bbox_df[temp_bbox_df.ImageID == img_id]
img = read_url_image(temp_urls[i])



plot_image_bounding_box(img_array = img,
                        xmin = list(img_bbox['XMin']),
                        xmax = list(img_bbox['XMax']),
                        ymin = list(img_bbox['YMin']),
                        ymax = list(img_bbox['YMax']))






