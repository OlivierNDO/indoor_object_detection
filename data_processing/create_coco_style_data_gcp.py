### Overview
###############################################################################

### Configuration
###############################################################################
# Import Python Modules
import collections
import datetime
from google.cloud import storage
from io import BytesIO, StringIO
from operator import itemgetter
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
import numpy as np
import os
import pandas as pd
import pickle
from PIL import Image
import random
import requests
import tempfile
from tensorflow.keras.preprocessing.image import load_img
import time
import scipy
from sklearn.model_selection import train_test_split
import skimage
from skimage.transform import resize
import tqdm

# Tensorflow / Keras Modules
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization, GlobalAvgPool2D
from tensorflow.keras.layers import Add, ZeroPadding2D, AveragePooling2D, GaussianNoise, SeparableConv2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.layers import add
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Import Project Modules
from src import config_data_processing as cdp
from src import image_manipulation as imm
from src import misc_functions as mf
from src import modeling as m


intmd_save_loc = 'D:/iod_25_class_intmd_save/'
dict_save_loc = 'D:/iod_dict_write/'
dict_save_name = 'object_dict_list.pkl'





### Define Functions and Classes
###############################################################################

def resize_image_and_bounding_box(image, new_size, bounding_box_list):
    """
    Resize an image and list of bounding box coordinates at the same time.
    Note that new_size must be a single integer.
    """
    h, w, c = image.shape
    new_bbox_list = []
    for bounding_box in bounding_box_list:
        xmin, xmax, ymin, ymax = bounding_box
        x_scale = new_size / w
        y_scale = new_size / h
        box_width = xmax - xmin
        box_height = ymax - ymin
        new_xmin = int(xmin * (new_size / w))
        new_xmax = int(new_xmin + (box_width * x_scale))
        new_ymin = int(ymin * (new_size / h))
        new_ymax = int(new_ymin + (box_height * y_scale))
        new_bbox_list.append([new_xmin, new_xmax, new_ymin, new_ymax])
    
    resized_image = resize(image, (new_size, new_size))
    
    return resized_image, new_bbox_list




class DetectionImageRetriever:
    """
    Retrieve and process bounding box and image data for a specific class
    
    Args:
        class_name (str): class name corresponding to subset of images
    """
    
    def __init__(self, 
                 class_name,
                 save_loc = None,
                 local_gcs_json_path = f'{cdp.config_gcs_auth_json_path}',
                 image_id_col = 'ImageID',
                 bucket_name = f'{cdp.config_source_bucket_name}',
                 processed_bucket_subfolder = f'{cdp.config_processed_bucket_subfolder}',
                 processed_array_save_name = 'train_images_cropped.npy',
                 resize_height = cdp.config_resize_height,
                 resize_width = cdp.config_resize_width,
                 max_images = 5000
                 ):
        # Initialize Arguments
        self.class_name = class_name
        self.save_loc = save_loc
        self.local_gcs_json_path = local_gcs_json_path
        self.image_id_col = image_id_col
        self.local_gcs_json_path = local_gcs_json_path
        self.bucket_name = bucket_name
        self.processed_bucket_subfolder = processed_bucket_subfolder
        self.processed_array_save_name = processed_array_save_name
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.max_images = max_images
        
        # Reference Google Cloud Authentication Document
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.local_gcs_json_path
        
    def get_whole_images_and_bbox(self):
        # Retrieve Class Metadata
        image_retriever = imm.OpenCVImageClassRetriever(class_name = self.class_name)
        bbox_df = image_retriever.get_bounding_box_df()
        desc_df = image_retriever.get_class_desc_df()
        
        # Image IDs
        unique_img_ids = list(np.unique(bbox_df[self.image_id_col].values.tolist()))
        if self.max_images is not None:
            unique_img_ids = unique_img_ids[:self.max_images]
        
        # Read and Crop Images with Bounding Boxes
        img_coord_dict = {}
        img_array_dict = {}
        
        
        for img_id in tqdm.tqdm(unique_img_ids):
            try:
                # Subset Info Dataframes for Image ID
                bbox_df_i = bbox_df[bbox_df.ImageID == img_id]
                desc_df_i = desc_df[desc_df.ImageID == img_id]
                
                # Read Image
                img_i = imm.read_url_image(desc_df_i['OriginalURL'].values[0])
                h, w, c = img_i.shape
            
                # Read Coordinates and Convert to Integers
                bbox_coords = bbox_df_i[['XMin', 'XMax', 'YMin', 'YMax']].values.tolist()
                bbox_coord_ints = []
                for bb in bbox_coords:
                    xmin = int(bb[0] * w)
                    xmax = int(bb[1] * w)
                    ymin = int(bb[2] * h)
                    ymax = int(bb[3] * h)
                    bbox_coord_ints.append([xmin, xmax, ymin, ymax])
                
                
                
                img_resized, resized_coords = resize_image_and_bounding_box(image = img_i,
                                                                              new_size = self.resize_height,
                                                                              bounding_box_list = bbox_coord_ints)
                
                correct_shape = (self.resize_width, self.resize_height, 3)
                if (not imm.is_blank_img(img_resized) and img_resized.shape == correct_shape):
                    img_array_dict[img_id] = img_resized
                    img_coord_dict[img_id] = resized_coords
            except:
                pass
        return img_coord_dict, img_array_dict
    
    

def plot_image_bounding_box(img_arr, coords, labels,
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
    for i, x in enumerate(coords):
        xmin_p = x[0]
        xmax_p = x[1]
        ymin_p = x[2]
        ymax_p = x[3]
        box_width = xmax_p - xmin_p
        box_height = ymax_p - ymin_p
    
        # Create rectangle and label text
        rect = patches.Rectangle((xmin_p, ymin_p), box_width, box_height, linewidth = linewidth, edgecolor = box_color, facecolor = 'none')
        ax.text(xmin_p, ymin_p + y_offset, labels[i], color = text_color, fontsize = fontsize)
        ax.add_patch(rect)
    plt.imshow(img_arr)
    plt.show()


### Data Processing: Read & REsize Images, Get Bounding Box Coordinates
###############################################################################
image_id_list_dict = {}
coord_list_dict = {}
od_classes = cdp.config_obj_detection_classes[0:2]


for i, odc in enumerate(od_classes):
    mf.print_timestamp_message(f'Starting {odc} class {(i+1)} of {len(od_classes)}')
    image_retriever = DetectionImageRetriever(class_name = odc, max_images = 10, resize_height = 416, resize_width = 416)
    img_coord_dict, img_array_dict = image_retriever.get_whole_images_and_bbox()
    img_id_list = list(img_coord_dict.keys())
    image_id_list_dict[odc] = img_id_list
    for i, x in tqdm.tqdm(enumerate(img_id_list)):
        img_save_name = f'{intmd_save_loc}{x}.jpeg'
        im = Image.fromarray((img_array_dict.get(x) * 255).astype(np.uint8))
        im.save(img_save_name)
        
    coord_list_dict[odc] = img_coord_dict
    del img_id_list, img_coord_dict, img_array_dict;




  
### Create Dictionary
###############################################################################
unique_image_ids = list(set(mf.unnest_list_of_lists([list(x.keys()) for x in coord_list_dict.values()])))
unique_classes = od_classes
image_dict_list = []

for uii in tqdm.tqdm(unique_image_ids):
    object_list = []
    for uc in unique_classes:
        if uii in image_id_list_dict.get(uc):
            img_save_name = f'{intmd_save_loc}{uii}.jpeg'
            # Append Coordinates
            coords = coord_list_dict.get(uc).get(uii)
            for c in coords:
                obj_dict = {'name' : uc,
                            'xmin': int(c[0]),
                            'ymin': int(c[2]),
                            'xmax': int(c[1]),
                            'ymax': int(c[3])}
                object_list.append(obj_dict)
        image_dict = {'object' : object_list,
                      'filename' : img_save_name,
                      'width' : 416,
                      'height' : 416}
    image_dict_list.append(image_dict)
    


with open(f'{dict_save_loc}{dict_save_name}', 'wb') as f:
    pickle.dump(image_dict_list, f)
    


"""
Testing:


rand_i = random.choice(range(len(image_dict_list)))

temp = image_dict_list[rand_i]

temp_image = np.asarray(load_img(temp.get('filename')))
temp_coords = [[temp.get('object')[x].get('xmin'), temp.get('object')[x].get('xmax'),
               temp.get('object')[x].get('ymin'), temp.get('object')[x].get('ymax')] for x in range(len(temp.get('object')))]

temp_labels = [temp.get('object')[x].get('name') for x in range(len(temp.get('object')))]

plot_image_bounding_box(img_arr = temp_image,
                        coords = temp_coords,
                        labels = temp_labels,
                            box_color = 'red', text_color = 'red', 
                            fontsize = 11, linewidth = 1, y_offset = -10)

"""