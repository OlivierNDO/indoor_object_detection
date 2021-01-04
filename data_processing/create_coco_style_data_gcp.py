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


### Data Processing: Read & REsize Images, Get Bounding Box Coordinates
###############################################################################

image_id_list_dict = {}
coord_list_dict = {}
image_size_list_dict = {}

for i, odc in enumerate(cdp.config_obj_detection_classes[:2]):
    mf.print_timestamp_message(f'Starting {odc} class {(i+1)} of {len(cdp.config_obj_detection_classes)}')
    image_retriever = imm.OpenCVCroppedImageRetriever(class_name = odc,
                                                      max_images = 5,
                                                      resize_height = 416,
                                                      resize_width = 416)
    img_id_list, coord_list, img_arr, img_size_list = image_retriever.get_whole_images_and_bbox()
    for i, x in tqdm.tqdm(enumerate(img_id_list)):
        img_save_name = f'{intmd_save_loc}{x}.jpeg'
        im = Image.fromarray((img_arr[i] * 255).astype(np.uint8))
        im.save(img_save_name)
    image_id_list_dict[odc] = img_id_list
    coord_list_dict[odc] = coord_list
    image_size_list_dict[odc] = img_size_list
    del img_id_list, coord_list, img_arr;
    
    
### Create Dictionary
###############################################################################
    
unique_image_ids = list(set(mf.unnest_list_of_lists(list(image_id_list_dict.values()))))
unique_classes = list(image_id_list_dict.keys())
image_dict_list = []

for uii in tqdm.tqdm(unique_image_ids):
    object_list = []
    for uc in unique_classes:
        if uii in image_id_list_dict.get(uc):
            index_pos = [i for i, x in enumerate(image_id_list_dict.get(uc)) if x == uii]
            img_save_name = f'{intmd_save_loc}{x}.jpeg'
            # Append Coordinates
            coords = [coord_list_dict.get(uc)[i] for i in index_pos]
            for c in coords:
                obj_dict = {'name' : uc,
                            'xmin': int(c[0]),
                            'ymin': int(c[2]),
                            'xmax': int(c[1]),
                            'ymax': int(c[3])}
                #obj_dict = {'name' : uc,
                #            'xmin': int(c[0] * cdp.config_resize_width),
                #            'ymin': int(c[2] * cdp.config_resize_height),
                #            'xmax': int(c[1] * cdp.config_resize_width),
                #            'ymax': int(c[3] * cdp.config_resize_height)}
                object_list.append(obj_dict)
    image_dict = {'object' : object_list,
                  'filename' : img_save_name,
                  'width' : cdp.config_resize_width,
                  'height' : cdp.config_resize_height}
    image_dict_list.append(image_dict)
    
    
with open(f'{dict_save_loc}{dict_save_name}', 'wb') as f:
    pickle.dump(image_dict_list, f)
    
    
    
    
    
    
    
    


temp_coords = coord_list_dict.get('Sports equipment')[10]


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


plot_image_bounding_box(img_arr = img_arr[10], coords = [temp_coords], labels = ['label'],
                            box_color = 'red', text_color = 'red', 
                            fontsize = 11, linewidth = 1, y_offset = -10)






# Read and Save
table_image_retriever = imm.OpenCVCroppedImageRetriever(class_name = 'Kitchen & dining room table', max_images = 5000, resize_height = 416, resize_width = 416, save_loc = intmd_save_loc)
table_image_retriever.save_whole_images_and_bbox()


















