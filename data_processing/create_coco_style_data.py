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


### Data Processing: Read Whole Images and Boundign Boxes
###############################################################################
# Television
tv_image_retriever = imm.OpenCVCroppedImageRetriever(class_name = 'Television', max_images = 3000)
tv_img_id_list, tv_coord_list, tv_img_arr = tv_image_retriever.get_whole_images_and_bbox()

# Couch
couch_image_retriever = imm.OpenCVCroppedImageRetriever(class_name = 'Couch', max_images = 3000)
couch_img_id_list, couch_coord_list, couch_img_arr = couch_image_retriever.get_whole_images_and_bbox()


# Coffee Table
ct_image_retriever = imm.OpenCVCroppedImageRetriever(class_name = 'Coffee table', max_images = 3000)
ct_img_id_list, ct_coord_list, ct_img_arr = ct_image_retriever.get_whole_images_and_bbox()

# Piana
piano_image_retriever = imm.OpenCVCroppedImageRetriever(class_name = 'Piano', max_images = 3000)
piano_img_id_list, piano_coord_list, piano_img_arr = piano_image_retriever.get_whole_images_and_bbox()


# Bed
bed_image_retriever = imm.OpenCVCroppedImageRetriever(class_name = 'Bed', max_images = 3000)
bed_img_id_list, bed_coord_list, bed_img_arr = bed_image_retriever.get_whole_images_and_bbox()





### Write Files Using Yolo Formatting
###############################################################################



# Dictionaries with Objects to Reformat and Save
image_id_list_dict = {'Television' : tv_img_id_list,
                      'Couch' : couch_img_id_list,
                      'Coffee table' : ct_img_id_list,
                      'Piano' : piano_img_id_list,
                      'Bed' : bed_img_id_list}

coord_list_dict = {'Television' : tv_coord_list,
                   'Couch' : couch_coord_list,
                   'Coffee table' : ct_coord_list,
                   'Piano' : piano_coord_list,
                   'Bed' : bed_coord_list}

img_array_dict = {'Television' : tv_img_arr,
                   'Couch' : couch_img_arr,
                   'Coffee table' : ct_img_arr,
                   'Piano' : piano_img_arr,
                   'Bed' : bed_img_arr}

# Folders to Save Objects
img_write_folder = 'D:/iod_yolo_data/train/'
dict_write_folder = 'D:/iod_yolo_data/pascal_format/'
dict_list_save_name = 'object_dict_list.txt'

# Loop Over Unique Image IDs
unique_image_ids = list(set(mf.unnest_list_of_lists(list(image_id_list_dict.values()))))
unique_classes = list(image_id_list_dict.keys())

image_dict_list = []

for uii in tqdm.tqdm(unique_image_ids):
    object_list = []
    for uc in unique_classes:
        if uii in image_id_list_dict.get(uc):
            index_pos = [i for i, x in enumerate(image_id_list_dict.get(uc)) if x == uii]
            # Save Image Array
            img_arr = img_array_dict.get(uc)[index_pos[0]]
            img_save_name = f'{img_write_folder}{uii}.jpeg'
            im = Image.fromarray((img_arr * 255).astype(np.uint8))
            im.save(img_save_name)
            
            # Append Coordinates
            coords = [coord_list_dict.get(uc)[i] for i in index_pos]
            for c in coords:
                obj_dict = {'name' : uc,
                            'xmin': int(c[0] * img_arr.shape[0]),
                            'ymin': int(c[2] * img_arr.shape[1]),
                            'xmax': int(c[1] * img_arr.shape[0]),
                            'ymax': int(c[3] * img_arr.shape[0])}
                object_list.append(obj_dict)
            width = img_arr.shape[0]
            height = img_arr.shape[1]
    image_dict = {'object' : object_list,
                  'filename' : img_save_name,
                  'width' : width,
                  'height' : height}
    image_dict_list.append(image_dict)
            
# Save List
#with open(f'{dict_write_folder}{dict_list_save_name}.txt', 'wb') as fp:
#    pickle.dump(image_dict_list, fp)
    
    
with open('D:/iod_yolo_data/pascal_format/object_dict_list.pkl', 'wb') as f:
    pickle.dump(image_dict_list, f)
    
    
# How to read list   
with open('D:/iod_yolo_data/pascal_format/object_dict_list.pkl', 'rb') as fp:
    use_dict = pickle.load(fp)   
    
    
    
    
    










