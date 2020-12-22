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

table_image_retriever = imm.OpenCVCroppedImageRetriever(class_name = 'Table', max_images = 5000, resize_height = 416, resize_width = 416)
table_img_id_list, table_coord_list, table_img_arr = table_image_retriever.get_whole_images_and_bbox()

chair_image_retriever = imm.OpenCVCroppedImageRetriever(class_name = 'Chair', max_images = 5000, resize_height = 416, resize_width = 416)
chair_img_id_list, chair_coord_list, chair_img_arr = chair_image_retriever.get_whole_images_and_bbox()

sculpture_image_retriever = imm.OpenCVCroppedImageRetriever(class_name = 'Sculpture', max_images = 5000, resize_height = 416, resize_width = 416)
sculpture_img_id_list, sculpture_coord_list, sculpture_img_arr = sculpture_image_retriever.get_whole_images_and_bbox()

laptop_image_retriever = imm.OpenCVCroppedImageRetriever(class_name = 'Laptop', max_images = 5000, resize_height = 416, resize_width = 416)
laptop_img_id_list, laptop_coord_list, laptop_img_arr = laptop_image_retriever.get_whole_images_and_bbox()

desk_image_retriever = imm.OpenCVCroppedImageRetriever(class_name = 'Desk', max_images = 5000, resize_height = 416, resize_width = 416)
desk_img_id_list, desk_coord_list, desk_img_arr = desk_image_retriever.get_whole_images_and_bbox()

houseplant_image_retriever = imm.OpenCVCroppedImageRetriever(class_name = 'Houseplant', max_images = 5000, resize_height = 416, resize_width = 416)
houseplant_img_id_list, houseplant_coord_list, houseplant_img_arr = houseplant_image_retriever.get_whole_images_and_bbox()

shelf_image_retriever = imm.OpenCVCroppedImageRetriever(class_name = 'Shelf', max_images = 5000, resize_height = 416, resize_width = 416)
shelf_img_id_list, shelf_coord_list, shelf_img_arr = shelf_image_retriever.get_whole_images_and_bbox()

couch_image_retriever = imm.OpenCVCroppedImageRetriever(class_name = 'Couch', max_images = 5000, resize_height = 416, resize_width = 416)
couch_img_id_list, couch_coord_list, couch_img_arr = couch_image_retriever.get_whole_images_and_bbox()

stairs_image_retriever = imm.OpenCVCroppedImageRetriever(class_name = 'Stairs', max_images = 5000, resize_height = 416, resize_width = 416)
stairs_img_id_list, stairs_coord_list, stairs_img_arr = stairs_image_retriever.get_whole_images_and_bbox()

vase_image_retriever = imm.OpenCVCroppedImageRetriever(class_name = 'Vase', max_images = 5000, resize_height = 416, resize_width = 416)
vase_img_id_list, vase_coord_list, vase_img_arr = vase_image_retriever.get_whole_images_and_bbox()

bench_image_retriever = imm.OpenCVCroppedImageRetriever(class_name = 'Bench', max_images = 5000, resize_height = 416, resize_width = 416)
bench_img_id_list, bench_coord_list, bench_img_arr = bench_image_retriever.get_whole_images_and_bbox()

computer_monitor_image_retriever = imm.OpenCVCroppedImageRetriever(class_name = 'Computer monitor', max_images = 5000, resize_height = 416, resize_width = 416)
computer_monitor_img_id_list, computer_monitor_coord_list, computer_monitor_img_arr = computer_monitor_image_retriever.get_whole_images_and_bbox()

computer_keyboard_image_retriever = imm.OpenCVCroppedImageRetriever(class_name = 'Computer keyboard', max_images = 5000, resize_height = 416, resize_width = 416)
computer_keyboard_img_id_list, computer_keyboard_coord_list, computer_keyboard_img_arr = computer_keyboard_image_retriever.get_whole_images_and_bbox()

sink_image_retriever = imm.OpenCVCroppedImageRetriever(class_name = 'Sink', max_images = 5000, resize_height = 416, resize_width = 416)
sink_img_id_list, sink_coord_list, sink_img_arr = sink_image_retriever.get_whole_images_and_bbox()

bed_image_retriever = imm.OpenCVCroppedImageRetriever(class_name = 'Bed', max_images = 5000, resize_height = 416, resize_width = 416)
bed_img_id_list, bed_coord_list, bed_img_arr = bed_image_retriever.get_whole_images_and_bbox()

cabinetry_image_retriever = imm.OpenCVCroppedImageRetriever(class_name = 'Cabinetry', max_images = 5000, resize_height = 416, resize_width = 416)
cabinetry_img_id_list, cabinetry_coord_list, cabinetry_img_arr = cabinetry_image_retriever.get_whole_images_and_bbox()

television_image_retriever = imm.OpenCVCroppedImageRetriever(class_name = 'Television', max_images = 5000, resize_height = 416, resize_width = 416)
television_img_id_list, television_coord_list, television_img_arr = television_image_retriever.get_whole_images_and_bbox()

curtain_image_retriever = imm.OpenCVCroppedImageRetriever(class_name = 'Curtain', max_images = 5000, resize_height = 416, resize_width = 416)
curtain_img_id_list, curtain_coord_list, curtain_img_arr = curtain_image_retriever.get_whole_images_and_bbox()

piano_image_retriever = imm.OpenCVCroppedImageRetriever(class_name = 'Piano', max_images = 5000, resize_height = 416, resize_width = 416)
piano_img_id_list, piano_coord_list, piano_img_arr = piano_image_retriever.get_whole_images_and_bbox()

mirror_image_retriever = imm.OpenCVCroppedImageRetriever(class_name = 'Mirror', max_images = 5000, resize_height = 416, resize_width = 416)
mirror_img_id_list, mirror_coord_list, mirror_img_arr = mirror_image_retriever.get_whole_images_and_bbox()

countertop_image_retriever = imm.OpenCVCroppedImageRetriever(class_name = 'Countertop', max_images = 5000, resize_height = 416, resize_width = 416)
countertop_img_id_list, countertop_coord_list, countertop_img_arr = countertop_image_retriever.get_whole_images_and_bbox()

drawer_image_retriever = imm.OpenCVCroppedImageRetriever(class_name = 'Drawer', max_images = 5000, resize_height = 416, resize_width = 416)
drawer_img_id_list, drawer_coord_list, drawer_img_arr = drawer_image_retriever.get_whole_images_and_bbox()

lamp_image_retriever = imm.OpenCVCroppedImageRetriever(class_name = 'Lamp', max_images = 5000, resize_height = 416, resize_width = 416)
lamp_img_id_list, lamp_coord_list, lamp_img_arr = lamp_image_retriever.get_whole_images_and_bbox()

fireplace_image_retriever = imm.OpenCVCroppedImageRetriever(class_name = 'Fireplace', max_images = 5000, resize_height = 416, resize_width = 416)
fireplace_img_id_list, fireplace_coord_list, fireplace_img_arr = fireplace_image_retriever.get_whole_images_and_bbox()

chest_of_drawers_image_retriever = imm.OpenCVCroppedImageRetriever(class_name = 'Chest of drawers', max_images = 5000, resize_height = 416, resize_width = 416)
chest_of_drawers_img_id_list, chest_of_drawers_coord_list, chest_of_drawers_img_arr = chest_of_drawers_image_retriever.get_whole_images_and_bbox()


### Write Files Using Yolo Formatting
###############################################################################



# Dictionaries with Objects to Reformat and Save
image_id_list_dict = {'Table' : table_img_id_list, 
                      'Chair' : chair_img_id_list, 
                      'Sculpture' : sculpture_img_id_list, 
                      'Laptop' : laptop_img_id_list, 
                      'Desk' : desk_img_id_list, 
                      'Houseplant' : houseplant_img_id_list, 
                      'Shelf' : shelf_img_id_list, 
                      'Couch' : couch_img_id_list, 
                      'Stairs' : stairs_img_id_list, 
                      'Vase' : vase_img_id_list, 
                      'Bench' : bench_img_id_list, 
                      'Computer monitor' : computer_monitor_img_id_list, 
                      'Computer keyboard' : computer_keyboard_img_id_list, 
                      'Sink' : sink_img_id_list, 
                      'Bed' : bed_img_id_list, 
                      'Cabinetry' : cabinetry_img_id_list, 
                      'Television' : television_img_id_list, 
                      'Curtain' : curtain_img_id_list, 
                      'Piano' : piano_img_id_list, 
                      'Mirror' : mirror_img_id_list, 
                      'Countertop' : countertop_img_id_list, 
                      'Drawer' : drawer_img_id_list, 
                      'Lamp' : lamp_img_id_list, 
                      'Fireplace' : fireplace_img_id_list, 
                      'Chest of drawers' : chest_of_drawers_img_id_list}

coord_list_dict = {'Table' : table_coord_list, 
                   'Chair' : chair_coord_list, 
                   'Sculpture' : sculpture_coord_list, 
                   'Laptop' : laptop_coord_list, 
                   'Desk' : desk_coord_list, 
                   'Houseplant' : houseplant_coord_list, 
                   'Shelf' : shelf_coord_list, 
                   'Couch' : couch_coord_list, 
                   'Stairs' : stairs_coord_list, 
                   'Vase' : vase_coord_list, 
                   'Bench' : bench_coord_list, 
                   'Computer monitor' : computer_monitor_coord_list, 
                   'Computer keyboard' : computer_keyboard_coord_list, 
                   'Sink' : sink_coord_list, 
                   'Bed' : bed_coord_list, 
                   'Cabinetry' : cabinetry_coord_list, 
                   'Television' : television_coord_list, 
                   'Curtain' : curtain_coord_list, 
                   'Piano' : piano_coord_list, 
                   'Mirror' : mirror_coord_list, 
                   'Countertop' : countertop_coord_list, 
                   'Drawer' : drawer_coord_list, 
                   'Lamp' : lamp_coord_list, 
                   'Fireplace' : fireplace_coord_list, 
                   'Chest of drawers' : chest_of_drawers_coord_list}

img_array_dict = {'Table' : table_img_arr, 
                  'Chair' : chair_img_arr, 
                  'Sculpture' : sculpture_img_arr, 
                  'Laptop' : laptop_img_arr, 
                  'Desk' : desk_img_arr, 
                  'Houseplant' : houseplant_img_arr, 
                  'Shelf' : shelf_img_arr, 
                  'Couch' : couch_img_arr, 
                  'Stairs' : stairs_img_arr, 
                  'Vase' : vase_img_arr, 
                  'Bench' : bench_img_arr, 
                  'Computer monitor' : computer_monitor_img_arr, 
                  'Computer keyboard' : computer_keyboard_img_arr, 
                  'Sink' : sink_img_arr, 
                  'Bed' : bed_img_arr, 
                  'Cabinetry' : cabinetry_img_arr, 
                  'Television' : television_img_arr, 
                  'Curtain' : curtain_img_arr, 
                  'Piano' : piano_img_arr, 
                  'Mirror' : mirror_img_arr, 
                  'Countertop' : countertop_img_arr, 
                  'Drawer' : drawer_img_arr, 
                  'Lamp' : lamp_img_arr, 
                  'Fireplace' : fireplace_img_arr, 
                  'Chest of drawers' : chest_of_drawers_img_arr}

# Folders to Save Objects
img_write_folder = 'C:/local_train_25/'
dict_write_folder = 'C:/dict_list_25/'
dict_list_save_name = 'object_dict_list.pkl'

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
    
    
with open(f'{dict_write_folder}{dict_list_save_name}', 'wb') as f:
    pickle.dump(image_dict_list, f)
    
    
# How to read list   
with open(f'{dict_write_folder}{dict_list_save_name}', 'rb') as fp:
    use_dict = pickle.load(fp)   
    
    
    
    
    










