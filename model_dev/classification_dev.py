### Overview
###############################################################################
# Takes functions written for room classification project (src.modeling)
# and runs a basic classification task on mirrors vs. countertops


### Configuration
###############################################################################
# Import Python Modules
import collections
import datetime
from google.cloud import storage
from io import BytesIO, StringIO
from operator import itemgetter
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

# Import Project Modules
import src.config_data_processing as cdp
import src.image_manipulation as imm
import src.misc_functions as mf
import src.modeling as m


### Retrieve & Process Class Data
###############################################################################   
# Pull Countertop Data from GCS
use_class = 'Countertop'
image_retriever = imm.OpenCVImageClassRetriever(class_name = use_class)
train_x_countertop, train_y_countertop = image_retriever.get_training_data()


# Pull Mirror Data from GCS
use_class = 'Mirror'
image_retriever = imm.OpenCVImageClassRetriever(class_name = use_class)
train_x_mirror, train_y_mirror = image_retriever.get_training_data()


### Combine and Prep Class Data
###############################################################################
# Combine into X and Y Arrays
y = ['Countertop' for x in range(train_x_countertop.shape[0])] + ['Mirror' for x in range(train_x_mirror.shape[0])]
x = np.vstack([train_x_countertop, train_x_mirror])

# Shuffle Randomly
x, y = m.shuffle_two_lists(x, y)

# Split into Train, Test, and Validation
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.4, random_state = 9242020)
valid_x, test_x, valid_y, test_y = train_test_split(x, y, test_size = 0.5, random_state = 9242020)

# Convert Lists to Arrays
train_x = np.array(train_x)
test_x = np.array(test_x)
valid_x = np.array(valid_x)

#train_y = np.array(train_y)
#test_y = np.array(test_y)
#valid_y = np.array(valid_y)

# Convert Strings in Y to Numbers
class_list, class_weights = m.make_class_weight_dict(list(train_y), return_dict = False)
class_list_int_dict = dict(zip(class_list, list(range(len(class_list)))))
train_y = np.vstack([class_list_int_dict.get(s) for s in train_y])
test_y = np.vstack([class_list_int_dict.get(s) for s in test_y])
valid_y = np.vstack([class_list_int_dict.get(s) for s in valid_y])




### Define Keras Configuration
###############################################################################

# Define Number of Steps Per Epoch (observations / batch size)
tsteps = int(train_x.shape[0]) // m.config_batch_size
vsteps = int(valid_x.shape[0]) // m.config_batch_size


train_gen = m.np_array_to_batch_gen_aug(train_x, train_y)
valid_gen = m.np_array_to_batch_gen(valid_x, valid_y)
test_gen = m.np_array_to_batch_gen(test_x, test_y)


lr_schedule = m.CyclicalRateSchedule(min_lr = m.config_min_lr, max_lr = m.config_max_lr,
                                   n_epochs = m.config_epochs,
                                   warmup_epochs = m.config_warmup_epochs,
                                   cooldown_epochs = m.config_cooldown_epochs,
                                   cycle_length = m.config_cycle_length,
                                   logarithmic = True,
                                   decrease_factor = 0.9)

lr_schedule.plot_cycle()

### Model Training
###############################################################################

# Start timer and clear session
train_start_time = time.time()
keras.backend.clear_session()
    
# Checkpoint and early stopping
check_point = keras.callbacks.ModelCheckpoint(m.config_model_save_name, monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'min')
early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss', mode = 'min',  patience = m.config_max_worse_epochs)
csv_name = "log_{ts}.csv".format(ts = m.config_model_timestamp)
csv_logger = keras.callbacks.CSVLogger(csv_name)

# Define model, scale to multiple GPUs, and start training
model = m.conv_10_layer()
model.compile(loss='categorical_crossentropy',
              optimizer = m.Adam(),
              metrics = ['categorical_accuracy'])

model.fit(train_gen,
          epochs = m.config_epochs,
          validation_data = valid_gen,
          validation_steps = vsteps,
          steps_per_epoch = tsteps,
          callbacks = [early_stop],
          #callbacks = [check_point, early_stop, lr_schedule.lr_scheduler(), csv_logger],
          class_weight = dict(zip(list(range(len(class_weights))), class_weights)))

train_end_time = time.time()
m.sec_to_time_elapsed(train_end_time, train_start_time)



















