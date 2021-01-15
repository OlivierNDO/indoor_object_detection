### Configuration - Packages
###############################################################################

from google.cloud import storage
import copy
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization, GlobalAvgPool2D, concatenate
from tensorflow.keras.layers import Add, ZeroPadding2D, AveragePooling2D, GaussianNoise, SeparableConv2D, Concatenate, LeakyReLU, Reshape, Lambda, Permute
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.layers import add
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
import sklearn
from skimage.transform import resize
import tensorflow as tf
import tqdm
import numpy as np
import pickle
from PIL import Image
import os
import random
from io import BytesIO
import io


# Import Project Modules
from src import config_data_processing as cdp
from src import image_manipulation as imm
from src import misc_functions as mf
from src import modeling as m
from src import loss_functions as lf

### Configuration - File Paths
###############################################################################
my_project_folder = 'D:/indoor_object_detection/'
model_save_path = cdp.config_model_save_folder
model_save_name = 'yolo_detector.hdf5'

# Path to Images (created in "create_coco_style_data_gcp.py)
local_image_folder = 'C:/local_images/'

# Path to Text File
dict_write_folder = f'{my_project_folder}data/processed_data/'
dict_list_save_name = 'object_dict_list.pkl'

# Path to Transfer Learning Weights
weights_path = f'{my_project_folder}weights/yolo-voc.weights'


    
def plot_actual_image_bounding_box(img_arr, coords, labels,
                            box_color = 'red', text_color = 'red', 
                            fontsize = 11, linewidth = 1, y_offset = -10):
    """
    Create a matplotlib image plot with one or more bounding boxes
    
    Args:
        img_array (numpy.array): numpy array of image
        xmin (list): list of x-minimum coordinates (expressed as percentages)
        xmax (list): list of x-maximum coordinates (expressed as percentages)
        ymin (list): list of y-minimum coordinates (expressed as percentages)
        ymax (list): list of y-maximum coordinates (expressed as percentages)
        label (list): list of bounding box labels
        box_color (str): color to use in bounding box edge (defaults to 'red')
        text_color (str): color to use in text label (defaults to 'red')
        fontsize (int): size to use for label font (defaults to 11)
        linewidth (int): size to use for box edge line width (defaults to 1)
        y_offset (int): how far to offset text label from upper-left corner of bounding box (defaults to -10)
    """
    # Extract image dimensions and create plot object
    h, w, c = img_arr.shape
    fig,ax = plt.subplots(1)
    
    # Extract coordinates and dimensions
    for i, x in enumerate(coords):
        xmin_p = x[0]
        xmax_p = x[1]
        ymin_p = x[2]
        ymax_p = x[3]
        box_width = xmax_p - xmin_p
        box_height = ymax_p - ymin_p
    
        # Create rectangle and label text
        rect = patches.Rectangle((xmin_p, ymin_p), box_width, box_height, linewidth = linewidth, edgecolor = box_color, facecolor = 'none')
        ax.text(xmin_p, ymin_p + y_offset, labels[i], color = text_color, fontsize = fontsize)
        ax.add_patch(rect)
    plt.imshow(img_arr)
    plt.show()




### Load Data
###############################################################################
# Load Data
with open(f'{dict_write_folder}{dict_list_save_name}', 'rb') as fp:
    train_image = pickle.load(fp)   
    
    
### Plot Bounding Boxes
###############################################################################
    
rand_i = random.choice(range(len(train_image)))

temp = train_image[rand_i]

temp_image = np.asarray(load_img(temp.get('filename')))
plt.imshow(temp_image)

temp_coords = [[temp.get('object')[x].get('xmin'), temp.get('object')[x].get('xmax'),
               temp.get('object')[x].get('ymin'), temp.get('object')[x].get('ymax')] for x in range(len(temp.get('object')))]

temp_labels = [temp.get('object')[x].get('name') for x in range(len(temp.get('object')))]

plot_actual_image_bounding_box(img_arr = temp_image,
                        coords = temp_coords,
                        labels = temp_labels,
                            box_color = 'red', text_color = 'red', 
                            fontsize = 11, linewidth = 1, y_offset = -10)



