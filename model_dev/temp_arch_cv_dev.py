### Overview
###############################################################################
# 3-Class Classfification with 19-layer CNN & Basic Image Augmentation
#
# Test Set Results
#   accuracy       true positive rate  true negative rate
#   0.90457             0.84879             0.93246

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

# Cropped 'Fireplace' Images
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
#train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.3, shuffle = True, random_state = 100)

# Split Test Set into 15/15 Test/Validation
#valid_x, test_x, valid_y, test_y = train_test_split(test_x, test_y, test_size = 0.5, shuffle = True, random_state = 100)


### Data Processing: Class Weights & Augmentation Generator
###############################################################################
# Class Weight Dictionary
#class_weight_dict = imm.make_class_weight_dict([np.argmax(x) for x in train_y], return_dict = True)




### Model Architectures to Compare
###############################################################################

# Learning Rate Schedule
lr_schedule = m.CyclicalRateSchedule(min_lr = 0.000015,
                                     max_lr = 0.00025,
                                     n_epochs = 400,
                                     warmup_epochs = 5,
                                     cooldown_epochs = 1,
                                     cycle_length = 10,
                                     logarithmic = True,
                                     decrease_factor = 0.9)

# Generator to Augment Images During Training
def image_flip_batch_generator(x_arr, y_arr, h_flip_every = 2, v_flip_every = 4, batch_size = 20):
    """
    Create Keras generator objects for minibatch training that flips images vertically and horizontally
    Args:
        x_arr: array of predictors
        y_arr: array of targets
        h_flip_every: every <n> images, flip horizontally
        v_flip_every: every <n> images, flip vertically
        batch_size: size of minibatches
    """
    indices = np.arange(len(x_arr)) 
    batch_list = []
    while True:
            np.random.shuffle(indices) 
            for i in indices:
                batch_list.append(i)
                if len(batch_list)==batch_size:
                    x_arr_aug = np.array([np.fliplr(x) if (j  % h_flip_every) == 0 else x for j, x in enumerate(x_arr[batch_list])])
                    x_arr_aug = np.array([np.flipud(x) if (j  % v_flip_every) == 0 else x for j, x in enumerate(x_arr_aug)])
                    yield x_arr_aug, y_arr[batch_list]
                    batch_list=[]
                    

# Functions to Make Deeper Networks w/ Fewer Lines of Code
def triple_conv2d(input_x, filter_size, kernel_size, activation):
    # Layer 1
    x = Conv2D(filters = filter_size, kernel_size = kernel_size, strides = (2,2), padding = 'same', use_bias = False)(input_x)
    x = Activation(activation)(x)
    x = BatchNormalization()(x)
    
    # Layer 2
    x = Conv2D(filters = filter_size, kernel_size = kernel_size, strides = (1,1), padding = 'same', use_bias = False)(x)
    x = Activation(activation)(x)
    x = BatchNormalization()(x)
    
    # Layer 3
    x = Conv2D(filters = filter_size, kernel_size = kernel_size, strides = (1,1), padding = 'same', use_bias = False)(x)
    x = Activation(activation)(x)
    x = BatchNormalization()(x)
    return x


def cnn_19_layer(n_classes, kernel_size, model_name = 'conv_19_layer', dense_dropout = 0.5, img_height = 200, img_width = 200, activ = 'relu'):
    
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
    x = Dense(n_classes, activation = "softmax")(x)
    
    # Model object
    model = Model(inputs = x_input, outputs = x, name = model_name) 
    return model


def cnn_10_layer(n_classes, kernel_size, model_name = 'conv_10_layer', dense_dropout = 0.5, img_height = 200, img_width = 200, activ = 'relu'):
    
    x_input = Input((img_height, img_width, 3))
    
    # Initial convolutional layer
    x = ZeroPadding2D(padding = (kernel_size, kernel_size))(x_input)
    x = Conv2D(50, (int(kernel_size * 2), int(kernel_size * 2)), strides = (1, 1), padding = 'valid', use_bias = False)(x)
    x = BatchNormalization()(x)
    x = Activation(activ)(x)
    
    
    # Triple convolutional layers
    x = triple_conv2d(x, filter_size = 50, kernel_size = (kernel_size,kernel_size), activation = activ)
    x = triple_conv2d(x, filter_size = 100, kernel_size = (kernel_size,kernel_size), activation = activ)
    x = triple_conv2d(x, filter_size = 100, kernel_size = (kernel_size,kernel_size), activation = activ)
    
    # Dense Layers
    x = Flatten()(x)
    x = Dense(100)(x)
    x = Activation(activ)(x)
    x = Dropout(dense_dropout)(x)
    x = Dense(50)(x)
    x = Activation(activ)(x)
    x = Dense(n_classes, activation = "softmax")(x)
    
    # Model object
    model = Model(inputs = x_input, outputs = x, name = model_name) 
    return model




class ArchitectureCrossValidator:
    """
    Perform cross validation on keras Model() objects with hyperparameters held constant
    Args:
        x (numpy.array): 4d array of images
        y (numpy.array): 2d array of binary classification indicators
        model_list (list): list of keras Model() objects with defined names
        lr_schedule (CyclicalRateSchedule): LR schedule object from CyclicalRateSchedule() class
        batch_generator (function): function that yields augmented batches
        k_folds (int): integer indicating number of folds to cross validate
        epochs (int): integer indicating number of epochs to train models for
        batch_size (int): batch size for model training
        optimizer (keras.Optimizer): optimizer to use in keras model compilation
        loss (string): loss to use in keras model compilation
        metrics (list): loss of strings to flow into keras metrics argument
    Returns:
        pandas.DataFrame() with fields 'model', 'fold', 'categorical_accuracy', 'execution_time'
    """
    
    def __init__(self, x, y, model_list, lr_schedule, batch_generator = image_flip_batch_generator,
                 k_folds = 10, epochs = 10, batch_size = 20, optimizer = Adam(),
                 loss = 'categorical_crossentropy', metrics = ['categorical_accuracy']):
        # Initialize Arguments
        self.x = x
        self.y = y
        self.model_list = model_list
        self.lr_schedule = lr_schedule
        self.batch_generator = batch_generator
        self.k_folds = k_folds
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
    
    def get_fold_indices(self):
        indices = list(range(0, self.x.shape[0]))
        random.shuffle(indices)
        n = max([1, math.ceil(len(indices) / self.k_folds)])
        return [indices[i:i+n] for i in range(0, len(indices), n)]
    
    def run_grid_search(self):
        output_categ_acc = []
        output_exec_time = []
        output_folds = []
        output_models = []
        output_model_number = []
        for iM, model in enumerate(self.model_list):
            for k in range(self.k_folds):
                # Separate Train and Test in Generators
                indices = self.get_fold_indices()
                train_i = mf.unnest_list_of_lists([j for i, j in enumerate(indices) if i != k])
                test_i = mf.unnest_list_of_lists([j for i, j in enumerate(indices) if i == k])
                train_gen = self.batch_generator(x[train_i], y[train_i], batch_size = self.batch_size)
                
                # Calculate Class Weights
                class_wt_dict = imm.make_class_weight_dict([np.argmax(x) for x in y[train_i]], return_dict = True)
                
                # Train Model
                train_start_time = time.time()
                keras.backend.clear_session()
                
                # Define Model Compilation
                model.compile(loss = self.loss,
                              optimizer = self.optimizer,
                              metrics = self.metrics)
                
                model.fit(train_gen,
                          epochs = self.epochs,
                          steps_per_epoch = int(len(train_i)) // self.batch_size,
                          callbacks = [self.lr_schedule],
                          class_weight = class_wt_dict)
                
                train_end_time = time.time()
                exec_time = train_end_time - train_start_time
                
    
                # Accuracy on Test Set
                pred_values = model.predict(x[test_i])
                output_categ_acc.append(np.mean(np.equal(np.argmax(y[test_i], axis=-1), np.argmax(pred_values, axis=-1))))                
                output_exec_time.append(exec_time)
                output_folds.append(k)
                output_models.append(model.name)
                output_model_number.append(iM)
                mf.print_timestamp_message(f'Completed fold {k+1} of {self.k_folds} for model {iM+1} of {len(self.model_list)}')
        # Collate Fold Results into DataFrame
        output_df = pd.DataFrame({'model' : output_models,
                                  'model_number' : output_model_number,
                                  'fold' : output_folds,
                                  'categorical_accuracy' : output_categ_acc,
                                  'execution_time' : output_exec_time})
        return output_df
            
        

cross_validator = ArchitectureCrossValidator(x = x, y = y,
                                             model_list = [cnn_10_layer(n_classes = 3, kernel_size = 3),
                                                           cnn_19_layer(n_classes = 3, kernel_size = 3)],
                                             lr_schedule = lr_schedule.lr_scheduler(),
                                             batch_generator = image_flip_batch_generator,
                                             k_folds = 10,
                                             epochs = 30)

arch_cv_results = cross_validator.run_grid_search()



