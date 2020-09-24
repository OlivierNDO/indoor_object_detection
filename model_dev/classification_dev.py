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
class_processor = imm.OpenCVMultiClassProcessor(class_list = ['Lamp', 'Studio couch', 'Chest of drawers'], max_images = 3000)


proc_data_dict = class_processor.get_train_test_valid_data()
train_x = proc_data_dict.get('TRAIN X')
test_x = proc_data_dict.get('TEST X')
valid_x = proc_data_dict.get('VALIDATION X')
train_y = proc_data_dict.get('TRAIN Y')
test_y = proc_data_dict.get('TEST Y')
valid_y = proc_data_dict.get('VALIDATION Y')
class_weight_dict = proc_data_dict.get('CLASS WEIGHT DICT')


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
model = m.resnet_conv_50_layer(n_classes = 3)
model.compile(loss='categorical_crossentropy',
              optimizer = m.Adam(),
              metrics = ['categorical_accuracy'])

model.fit(train_gen,
          epochs = m.config_epochs,
          validation_data = valid_gen,
          validation_steps = vsteps,
          steps_per_epoch = tsteps,
          callbacks = [check_point, early_stop],
          #callbacks = [check_point, early_stop, lr_schedule.lr_scheduler(), csv_logger],
          class_weight = class_weight_dict)

train_end_time = time.time()
m.sec_to_time_elapsed(train_end_time, train_start_time)

