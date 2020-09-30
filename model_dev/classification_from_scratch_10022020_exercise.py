### Overview
###############################################################################
# Takes functions written for room classification project (src.modeling)
# and runs a basic classification task on mirrors vs. countertops


# Next Steps (project):
#   Create folders with fewer imgaes per class
#   Add functionality to make classification only use subset of image with object
#   Use sliding window to turn classification model into object detection
#   https://www.pyimagesearch.com/2020/06/22/turning-any-cnn-image-classifier-into-an-object-detector-with-keras-tensorflow-and-opencv/
#
# Next Steps (knowledge sharing):
#   Build classification model from scratch rather than re-using models from previous G3
#
#


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
get_class1 = 'Chest of drawers'
x1 = mf.read_gcs_numpy_array(bucket_name = cdp.config_source_bucket_name, file_name = f'processed_files/{get_class1}/train_images_cropped.npy')
plt.imshow(x1[0])

# Cropped 'Chest of drawers' Images
get_class2 = 'Fireplace'
x2 = mf.read_gcs_numpy_array(bucket_name = cdp.config_source_bucket_name, file_name = f'processed_files/{get_class2}/train_images_cropped.npy')
plt.imshow(x2[0])

# Cropped 'Sofa bed' Images
get_class3 = 'Sofa bed'
x3 = mf.read_gcs_numpy_array(bucket_name = cdp.config_source_bucket_name, file_name = f'processed_files/{get_class3}/train_images_cropped.npy')
plt.imshow(x3[0])


### Data Processing: Create Y Variable
###############################################################################
# to be consistent, we should always order classes alphabetically

# Create and Combine Binary Variables
binary1 = np.array([[1, 0, 0]] * x1.shape[0])
binary2 = np.array([[0, 1, 0]] * x2.shape[0])
binary3 = np.array([[0, 0, 1]] * x3.shape[0])
y = np.vstack([binary1, binary2, binary3])
x = np.vstack([x1, x2, x3])

# Delete Objects from Memory
del x1; del x2; del x3; del binary1; del binary2; del binary3;


### Data Processing: Train, Test, Validation Split
###############################################################################
# Split X and Y into 70/30 Train/Test
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.3, shuffle = True, random_state = 100)

# Split Test Set into 15/15 Test/Validation
valid_x, test_x, valid_y, test_y = train_test_split(test_x, test_y, test_size = 0.5, shuffle = True, random_state = 100)


### Data Processing: Class Weights
###############################################################################
# From Scratch ... (fyi)
#classes = [np.argmax(x) for x in train_y]
#unique_classes = sorted(np.unique(classes))
#class_counts = [sum([1 for x in classes if x == uc]) for uc in unique_classes]
#class_weights = [max(class_counts) / x for x in class_counts]
#class_weight_dict = dict(zip(unique_classes, class_weights))

# Using Function
class_weight_dict = imm.make_class_weight_dict([np.argmax(x) for x in train_y], return_dict = True)


### Model Configuration
###############################################################################
# Parameters
mc_batch_size = 20
mc_epochs = 10
mc_learning_rate = 0.001
mc_dropout = 0.2

# Calculate Training Steps
tsteps = int(train_x.shape[0]) // mc_batch_size
vsteps = int(valid_x.shape[0]) // mc_batch_size


### Model Architecture
###############################################################################

# Clear Session (this removes any trained model from your PC's memory)
train_start_time = time.time()
keras.backend.clear_session()

# Define Shape of Input (width, height, channels)
x_input = Input((200, 200, 3))

# Convolutional Layer 1
# 3 x 3 filters are popular for reasons explained here:
#       https://towardsdatascience.com/deciding-optimal-filter-size-for-cnns-d6f7b56f9363
x = ZeroPadding2D(padding = (3, 3))(x_input)
x = Conv2D(50, (6, 6), strides = (1, 1), padding = 'valid', use_bias = False)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

# Convolutional Layer 2
x = Conv2D(50, (3, 3), strides = (2, 2), use_bias = False)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

# Convolutional Layer 3
x = Conv2D(50, (3, 3), strides = (2, 2), use_bias = False)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)


# Convolutional Layer 4
x = Conv2D(50, (3, 3), strides = (2, 2), use_bias = False)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)


# Convolutional Layer 5 (same as 4)
x = Conv2D(150, (3, 3), strides = (2, 2), use_bias = False)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

# Convolutional Layer 6
x = Conv2D(150, (3, 3), strides = (2, 2), use_bias = False)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

# Convolutional Layer 7
x = Conv2D(150, (3, 3), strides = (2, 2), use_bias = False)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size = (2,2), strides = (2, 2))(x)

# Dense Layers (output size 3 equal to number of classes)
x = Flatten()(x)
x = Dense(200)(x)
x = Dropout(mc_dropout)(x)
x = Dense(100)(x)
x = Activation('relu')(x)
x = Dense(3, activation = 'softmax')(x)

model = Model(inputs = x_input, outputs = x, name = 'cnn_from_scratch') 
model.summary()

### Model Fitting
###############################################################################


# Define model, scale to multiple GPUs, and start training
model.compile(loss='categorical_crossentropy',
              optimizer = Adam(learning_rate = mc_learning_rate),
              metrics = ['categorical_accuracy'])

model.fit(train_x, train_y,
          epochs = mc_epochs,
          validation_data = (valid_x, valid_y),
          steps_per_epoch = tsteps,
          validation_steps = vsteps,
          class_weight = class_weight_dict)

train_end_time = time.time()
m.sec_to_time_elapsed(train_end_time, train_start_time)







