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
get_class = 'Chest of drawers'

cropped_images = mf.read_gcs_numpy_array(bucket_name = cdp.config_source_bucket_name, file_name = f'processed_files/{get_class}/train_images_cropped.npy')
whole_images = mf.read_gcs_numpy_array(bucket_name = cdp.config_source_bucket_name, file_name = f'processed_files/{get_class}/train_images.npy')
bbox_df = mf.read_gcs_csv_to_pandas(bucket_name = cdp.config_source_bucket_name, file_name = f'processed_files/{get_class}/train_bbox.csv')
#train_class_df = mf.read_gcs_csv_to_pandas(bucket_name = cdp.config_source_bucket_name, file_name = f'processed_files/{get_class}/train_class_df.csv')
































