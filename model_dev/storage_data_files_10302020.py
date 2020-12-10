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
from PIL import Image
import random
import requests
import tempfile
from tensorflow.keras.preprocessing.image import load_img
import time
from sklearn.model_selection import train_test_split
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


### Data Processing: Read Cropped Images for Classification
###############################################################################

# Cropped 'Chest of drawers' Images
get_class = 'Sofa bed'
cropped_images = mf.read_gcs_numpy_array(bucket_name = cdp.config_source_bucket_name, file_name = f'processed_files/{get_class}/train_images_cropped.npy')
whole_images = mf.read_gcs_numpy_array(bucket_name = cdp.config_source_bucket_name, file_name = f'processed_files/{get_class}/train_images.npy')
bbox_df = mf.read_gcs_csv_to_pandas(bucket_name = cdp.config_source_bucket_name, file_name = f'processed_files/{get_class}/train_bbox.csv')
#train_class_df = mf.read_gcs_csv_to_pandas(bucket_name = cdp.config_source_bucket_name, file_name = f'processed_files/{get_class}/train_class_df.csv')




### Data Processing: Read Cropped Images for Classification
###############################################################################

get_class = 'Sofa bed'

image_retriever = imm.OpenCVCroppedImageRetriever(class_name = get_class, max_images = 100)

coord_list, img_arr = image_retriever.get_whole_images_and_bbox()









image_retriever.cropped_obj_images_to_gcs()




























plt.imshow(whole_images[0])


bbox_coords = bbox_df_i[['XMin', 'XMax', 'YMin', 'YMax']].values.tolist()

imm.plot_image_bounding_box(img_arr = whole_images[0],
                            xmin = [bbox_df.XMin[0]],
                            xmax = [bbox_df.XMax[0]],
                            ymin = [bbox_df.YMin[0]],
                            ymax = [bbox_df.YMax[0]],
                            label = [get_class],
                            box_color = 'red',
                            text_color = 'red', 
                            fontsize = 11,
                            linewidth = 1,
                            y_offset = -10)



img_sub = imm.extract_image_subset(img_arr = whole_images[0],
                                   xmin = bbox_df.XMin[0],
                                   xmax = bbox_df.XMax[0],
                                   ymin = bbox_df.YMin[0],
                                   ymax = bbox_df.YMax[0],
                                   decimal = True)

plt.imshow(img_sub)

xx = 0

plt.imshow(cropped_images[xx])

plt.imshow(whole_images[xx])

print(bbox_df.loc[xx])















