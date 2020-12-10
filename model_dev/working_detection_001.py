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
# Read Some Images, Coordinates, and Image IDs
get_class = 'Sofa bed'
image_retriever = imm.OpenCVCroppedImageRetriever(class_name = get_class, max_images = 100)
img_id_list, coord_list, img_arr = image_retriever.get_whole_images_and_bbox()


# Test a Few
plot_i = 20
imm.plot_image_bounding_box(img_arr = img_arr[plot_i],
                            xmin = [coord_list[plot_i][0]],
                            xmax = [coord_list[plot_i][1]],
                            ymin = [coord_list[plot_i][2]],
                            ymax = [coord_list[plot_i][3]],
                            label = [get_class],
                            box_color = 'red',
                            text_color = 'red', 
                            fontsize = 11,
                            linewidth = 1,
                            y_offset = -10)




### Data Processing: Train, Test, Validation Split
###############################################################################
# Define X and Y
x = img_arr
y = np.array(coord_list)

# Split X and Y into 70/30 Train/Test
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.3, shuffle = True, random_state = 100)

# Split Test Set into 15/15 Test/Validation
valid_x, test_x, valid_y, test_y = train_test_split(test_x, test_y, test_size = 0.5, shuffle = True, random_state = 100)


### Define Functions
###############################################################################
def IOU(bbox1, bbox2):
    """
    Calculate overlap between two bounding boxes - intersection area / unity area
    Args:
        bbox1 (list): list of four decimals between 0 and 1
        bbox2 (list): list of four decimals between 0 and 1
    Returns:
        float
    """
    x1, y1, w1, h1 = bbox1[0], bbox1[1], bbox1[2], bbox1[3]  # TODO: Check if its more performant if tensor elements are accessed directly below.
    x2, y2, w2, h2 = bbox2[0], bbox2[1], bbox2[2], bbox2[3]

    w_I = min(x1 + w1, x2 + w2) - max(x1, x2)
    h_I = min(y1 + h1, y2 + h2) - max(y1, y2)
    if w_I <= 0 or h_I <= 0:  # no overlap
        return 0
    I = w_I * h_I

    U = w1 * h1 + w2 * h2 - I

    return I / U

def dist(bbox1, bbox2):
    """
    Calculate distance between actual and expected bounding boxes
    Args:
        bbox1 (list): list of four decimals between 0 and 1
        bbox2 (list): list of four decimals between 0 and 1
    Returns:
        float
    """
    return np.sqrt(np.sum(np.square(bbox1[:2] - bbox2[:2])))

# Functions to Make Deeper Networks w/ Fewer Lines of Code
def triple_conv2d(input_x, filter_size, kernel_size, activation, use_batchnorm = True, use_maxpool = True):
    # Layer 1
    x = Conv2D(filters = filter_size, kernel_size = kernel_size, strides = (2,2), padding = 'same')(input_x)
    x = Activation(activation)(x)
    x = BatchNormalization()(x)
    
    # Layer 2
    x = Conv2D(filters = filter_size, kernel_size = kernel_size, strides = (1,1), padding = 'same')(x)
    x = Activation(activation)(x)
    x = BatchNormalization()(x)
    
    # Layer 3
    x = Conv2D(filters = filter_size, kernel_size = kernel_size, strides = (1,1), padding = 'same')(x)
    x = Activation(activation)(x)
    x = BatchNormalization()(x)
    return x


def cnn_19_layer_regression(output_shape, kernel_size, dense_dropout = 0.5, img_height = 200, img_width = 200, activ = 'relu'):
    
    x_input = Input((img_height, img_width, 3))
    
    # Initial convolutional layer
    x = ZeroPadding2D(padding = (kernel_size, kernel_size))(x_input)
    x = Conv2D(50, (int(kernel_size * 2), int(kernel_size * 2)), strides = (1, 1), padding = 'valid', use_bias = False)(x)
    x = BatchNormalization()(x)
    x = Activation(activ)(x)
    
    
    # Triple convolutional layers
    x = triple_conv2d(x, filter_size = 50, kernel_size = (kernel_size,kernel_size), activation = activ)
    x = triple_conv2d(x, filter_size = 100, kernel_size = (kernel_size,kernel_size), activation = activ)
    x = triple_conv2d(x, filter_size = 150, kernel_size = (kernel_size,kernel_size), activation = activ)
    x = triple_conv2d(x, filter_size = 200, kernel_size = (kernel_size,kernel_size), activation = activ)
    x = triple_conv2d(x, filter_size = 150, kernel_size = (kernel_size,kernel_size), activation = activ)
    x = triple_conv2d(x, filter_size = 100, kernel_size = (kernel_size,kernel_size), activation = activ)
    
    # Dense Layers
    x = Flatten()(x)
    x = Dense(100)(x)
    x = Activation(activ)(x)
    x = Dropout(dense_dropout)(x)
    x = Dense(50)(x)
    x = Activation(activ)(x)
    x = Dense(output_shape, activation = "softmax")(x)
    
    # Model object
    model = Model(inputs = x_input, outputs = x, name = 'conv_19_layer_regression') 
    return model    


def batch_generator(x_arr, y_arr, batch_size = 20):
    """
    Create Keras generator objects for minibatch training
    Args:
        x_arr: array of predictors
        y_arr: array of targets
        batch_size: size of minibatches
    """
    indices = np.arange(len(x_arr)) 
    batch_list = []
    while True:
            np.random.shuffle(indices) 
            for i in indices:
                batch_list.append(i)
                if len(batch_list)==batch_size:
                    yield x_arr[batch_list], y_arr[batch_list]
                    batch_list=[]



### Model Configuration
###############################################################################
# Parameters
mc_model_save_name = 'C:/keras_model_save/sofa_bed_detector.hdf5'
mc_csv_log_save_name = 'C:/keras_epoch_results/sofabed_epoch_results.csv'
mc_batch_size = 20
mc_epochs = 400
mc_learning_rate = 0.001
mc_dropout = 0.2

# Calculate Training Steps
tsteps = int(train_x.shape[0]) // mc_batch_size
vsteps = int(valid_x.shape[0]) // mc_batch_size

# Create Learning Rate Schedule
lr_schedule = m.CyclicalRateSchedule(min_lr = 0.000015,
                                     max_lr = 0.00025,
                                     n_epochs = 400,
                                     warmup_epochs = 5,
                                     cooldown_epochs = 1,
                                     cycle_length = 10,
                                     logarithmic = True,
                                     decrease_factor = 0.9)

lr_schedule.plot_cycle()

# Create Augmentation Generator Objects
train_gen = batch_generator(train_x, train_y, batch_size = mc_batch_size)
valid_gen = batch_generator(valid_x, valid_y, batch_size = mc_batch_size)
test_gen = batch_generator(test_x, test_y, batch_size = mc_batch_size)


### Model Architecture
###############################################################################
# Clear Session (this removes any trained model from your PC's memory)
train_start_time = time.time()
keras.backend.clear_session()


# 19-Layer CNN
model = cnn_19_layer_regression(output_shape = 4, kernel_size = 3)
#model.compile('adadelta', 'mse')

### Model Fitting
###############################################################################
# Keras Model Checkpoints (used for early stopping & logging epoch accuracy)
check_point = keras.callbacks.ModelCheckpoint(mc_model_save_name, monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'min')
early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss', mode = 'min',  patience = 15)
csv_logger = keras.callbacks.CSVLogger(mc_csv_log_save_name)


# Define Model Compilation
model.compile(loss='mse',
              optimizer = Adam(),
              metrics = ['accuracy'])

model.fit(train_gen,
          epochs = mc_epochs,
          validation_data = valid_gen,
          steps_per_epoch = tsteps,
          validation_steps = vsteps,
          callbacks = [check_point, early_stop, csv_logger, lr_schedule.lr_scheduler()])

train_end_time = time.time()
m.sec_to_time_elapsed(train_end_time, train_start_time)




epoch_results = pd.read_csv(m.config_csv_save_name)
plt.plot(epoch_results['epoch'], epoch_results['loss'])
plt.show()




"""
### Define and Fit Regression Model
###############################################################################
# Model Definition
model = cnn_19_layer_regression(output_shape = 4, kernel_size = 3)
model.compile('adadelta', 'mse')


# Training Loop
num_objects = 1
num_epochs_flipping = 50
num_epochs_no_flipping = 0  # has no significant effect
flipped_train_y = np.array(train_y)
flipped = np.zeros((len(flipped_train_y), num_epochs_flipping + num_epochs_no_flipping))
ious_epoch = np.zeros((len(flipped_train_y), num_epochs_flipping + num_epochs_no_flipping))
dists_epoch = np.zeros((len(flipped_train_y), num_epochs_flipping + num_epochs_no_flipping))
mses_epoch = np.zeros((len(flipped_train_y), num_epochs_flipping + num_epochs_no_flipping))

# TODO: Calculate ious directly for all samples (using slices of the array pred_y for x, y, w, h).
for epoch in range(num_epochs_flipping):
    print(f'Epoch {epoch}')
    model.fit(train_x, flipped_train_y, epochs = 1, validation_data=(valid_x, valid_y), verbose=2)
    pred_y = model.predict(train_x)

    for sample, (pred, exp) in enumerate(zip(pred_y, flipped_train_y)):
        
        # TODO: Make this simpler.
        pred = pred.reshape(num_objects, -1)
        exp = exp.reshape(num_objects, -1)
        
        pred_bboxes = pred[:, :4]
        exp_bboxes = exp[:, :4]
        
        # TODO: Try flipping array and see if results differ.
        ious = np.zeros((num_objects, num_objects))
        dists = np.zeros((num_objects, num_objects))
        mses = np.zeros((num_objects, num_objects))
        for i, exp_bbox in enumerate(exp_bboxes):
            for j, pred_bbox in enumerate(pred_bboxes):
                ious[i, j] = IOU(exp_bbox, pred_bbox)
                dists[i, j] = dist(exp_bbox, pred_bbox)
                mses[i, j] = np.mean(np.square(exp_bbox - pred_bbox))
                
        new_order = np.zeros(num_objects, dtype=int)
        
        for i in range(num_objects):
            # Find pred and exp bbox with maximum iou and assign them to each other (i.e. switch the positions of the exp bboxes in y).
            ind_exp_bbox, ind_pred_bbox = np.unravel_index(mses.argmin(), mses.shape)
            ious_epoch[sample, epoch] += ious[ind_exp_bbox, ind_pred_bbox]
            dists_epoch[sample, epoch] += dists[ind_exp_bbox, ind_pred_bbox]
            mses_epoch[sample, epoch] += mses[ind_exp_bbox, ind_pred_bbox]
            mses[ind_exp_bbox] = 1000000#-1  # set iou of assigned bboxes to -1, so they don't get assigned again
            mses[:, ind_pred_bbox] = 10000000#-1
            new_order[ind_pred_bbox] = ind_exp_bbox
        
        flipped_train_y[sample] = exp[new_order].flatten()
        
        flipped[sample, epoch] = 1. - np.mean(new_order == np.arange(num_objects, dtype=int))#np.array_equal(new_order, np.arange(num_objects, dtype=int))  # TODO: Change this to reflect the number of flips.
        ious_epoch[sample, epoch] /= num_objects
        dists_epoch[sample, epoch] /= num_objects
        mses_epoch[sample, epoch] /= num_objects
        
    mean_iou = np.mean(ious_epoch[:, epoch])
    mean_dist = np.mean(dists_epoch[:, epoch])
    mean_mse = np.mean(mses_epoch[:, epoch])
        
    print(f'Mean IOU: {mean_iou}')
    print(f'Mean dist: {mean_dist}')
    print(f'Mean mse: {mean_mse}')



"""
### Try Predictions
###############################################################################c


pred_test = model.predict(test_x)

























