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

# Tensorflow / Keras Modules
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Import Project Modules
import src.config_data_processing as cdp
import src.image_manipulation as imm
import src.misc_functions as mf


### Retrieve & Process Data: Countertops
###############################################################################   
# Pull Countertop Data from GCS
use_class = 'Countertop'
image_retriever = imm.OpenCVImageClassRetriever(class_name = use_class)
img_arr = image_retriever.get_class_image_array()
bbox_df = image_retriever.get_bounding_box_df()
desc_df = image_retriever.get_class_desc_df()


# Create a Dictionary: {ImageID : (numpy array, bounding box)}
bbox_df_copy = bbox_df.copy()

bbox_df_copy['XMin'] = [int(x * img_arr.shape[-2]) for x in bbox_df['XMin']]
bbox_df_copy['XMax'] = [int(x * img_arr.shape[-2]) for x in bbox_df['XMax']]
bbox_df_copy['YMin'] = [int(x * img_arr.shape[-2]) for x in bbox_df['YMin']]
bbox_df_copy['YMax'] = [int(x * img_arr.shape[-2]) for x in bbox_df['YMax']]


unique_image_ids = list(set(desc_df['ImageID']))
image_ids = list(bbox_df_copy['ImageID'])
y_array_list = mf.unnest_list_of_lists([bbox_df_copy[bbox_df_copy.ImageID == x][['XMin', 'XMax', 'YMin', 'YMax']].values.tolist() for x in unique_image_ids])
img_array_positions = mf.unnest_list_of_lists([[i for i,x in enumerate(unique_image_ids) if x == z] for z in image_ids])


train_x = np.array([img_arr[i] for i in img_array_positions])
train_y = np.array(y_array_list)







# train_imgs, classes_count, class_mapping = get_data(train_path)
# data_gen_train = get_anchor_gt(train_imgs, C, get_img_output_length, mode='train')
# calc_rpn


















#train_x = train_x.reshape(train_x.shape[0], -1)


train_x.shape[-1]
train_y.shape[-1]

#.reshape(train_x.shape[0], train_x.shape[-1])


#unique_image_ids[0]
#image_ids[image_ids == unique_image_ids[0]]
#image_ids[image_ids == '0000615b5a80f660']
#train_x = np.array([img_arr[i] for i in range(len(image_ids))])
#train_y = np.vstack(y_array_list)#.astype(np.float32)



# Build the model.
tf.keras.backend.clear_session()
model = keras.Sequential([
        #layers.Input(train_x.shape[1:]),
        layers.Dense(200, input_shape = train_x.shape), 
        layers.Activation('relu'), 
        layers.Dropout(0.2), 
        layers.Dense(train_y.shape[-1])
    ])
model.compile('adadelta', 'mse')



num_epochs = 5
for epoch in range(num_epochs):
    print(f'Epoch {epoch}')
    model.fit(train_x, train_y, epochs = 1, verbose=2)
    pred_y = model.predict(train_x)




