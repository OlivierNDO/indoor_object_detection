# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 11:19:50 2020

@author: user
"""

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


### Combine Class Data
###############################################################################
y = ['Countertop' for x in range(train_x_countertop.shape[0])] + ['Mirror' for x in range(train_x_mirror.shape[0])]
x = np.vstack([train_x_countertop, train_x_mirror])



tsteps = int(train_x.shape[0]) // m.config_batch_size
#vsteps = int(valid_x.shape[0]) // config_batch_size





class_list, class_weights = m.make_class_weight_dict(train_y, return_dict = False)
train_gen = m.np_array_to_batch_gen_aug(train_x, train_y)
#valid_gen = m.np_array_to_batch_gen(valid_x, valid_y)
#test_gen = np_array_to_batch_gen(test_x, test_y)


lr_schedule = m.CyclicalRateSchedule(min_lr = m.config_min_lr, max_lr = m.config_max_lr,
                                   n_epochs = m.config_epochs,
                                   warmup_epochs = m.config_warmup_epochs,
                                   cooldown_epochs = m.config_cooldown_epochs,
                                   cycle_length = m.config_cycle_length,
                                   logarithmic = True,
                                   decrease_factor = 0.9)

lr_schedule.plot_cycle()






















