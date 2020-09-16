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
import tempfile
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
                 resize_width = cdp.config_resize_width
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
        image_arrays = imm.load_resize_images_from_urls(url_list = urls, resize_height = self.resize_height, resize_width = self.resize_width)
        image_arrays_concat = np.array(image_arrays)
        
        # Write Images to Google Cloud Storage Bucket
        image_save_name = f'{self.processed_bucket_subfolder}{self.class_name}/{self.processed_array_save_name}'
        mf.print_timestamp_message(f'Writing images to GCS bucket/folder {self.bucket_name}/{image_save_name}')
        mf.save_np_array_to_gsc(np_array = image_arrays_concat, bucket_name = self.bucket_name, file_name = image_save_name)
        
        # Write Bounding Box Csv to Google Cloud Storage Bucket
        bbox_save_name = f'{self.processed_bucket_subfolder}{self.class_name}/{self.processed_bbox_save_name}'
        mf.print_timestamp_message(f'Writing bounding box csv file to GCS bucket/folder {self.bucket_name}/{bbox_save_name}')
        mf.write_csv_to_gcs(dframe = bbox_df, bucket_name = self.bucket_name, file_name = bbox_save_name)
        
        # Write Class Info Csv to Google Cloud Storage Bucket
        class_save_name = f'{self.processed_bucket_subfolder}{self.class_name}/{self.processed_class_save_name}'
        mf.print_timestamp_message(f'Writing class image info csv file to GCS bucket/folder {self.bucket_name}/{class_save_name}')
        mf.write_csv_to_gcs(dframe = class_image_df, bucket_name = self.bucket_name, file_name = class_save_name)
        
        

### Execute Functions
###############################################################################
image_retriever = OpenCVImageClassRetriever(class_name = 'Loveseat')
image_retriever.resize_and_save_images()



