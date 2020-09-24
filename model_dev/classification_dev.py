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


### Retrieve & Process Data: Countertops
###############################################################################   
# Pull Countertop Data from GCS
use_class = 'Countertop'
image_retriever = imm.OpenCVImageClassRetriever(class_name = use_class)
train_x_countertop, train_y_countertop = image_retriever.get_training_data()


# Pull Mirror Data from GCS
use_class = 'Mirror'
image_retriever = imm.OpenCVImageClassRetriever(class_name = use_class)
train_x_mirror, train_y_mirror = image_retriever.get_training_data()


























