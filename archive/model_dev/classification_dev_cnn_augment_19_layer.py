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
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.3, shuffle = True, random_state = 100)

# Split Test Set into 15/15 Test/Validation
valid_x, test_x, valid_y, test_y = train_test_split(test_x, test_y, test_size = 0.5, shuffle = True, random_state = 100)


### Data Processing: Class Weights & Augmentation Generator
###############################################################################
# Class Weight Dictionary
class_weight_dict = imm.make_class_weight_dict([np.argmax(x) for x in train_y], return_dict = True)


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


def cnn_19_layer(n_classes, kernel_size, dense_dropout = 0.5, img_height = 200, img_width = 200, activ = 'relu'):
    
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
    model = Model(inputs = x_input, outputs = x, name = 'conv_10_layer') 
    return model     



### Model Configuration
###############################################################################
# Parameters
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
train_gen = image_flip_batch_generator(train_x, train_y, batch_size = mc_batch_size)
valid_gen = image_flip_batch_generator(valid_x, valid_y, batch_size = mc_batch_size)
test_gen = image_flip_batch_generator(test_x, test_y, batch_size = mc_batch_size)

### Model Architecture
###############################################################################
# Clear Session (this removes any trained model from your PC's memory)
train_start_time = time.time()
keras.backend.clear_session()

# 19-Layer CNN
model = cnn_19_layer(n_classes = 3, kernel_size = 3)

### Model Fitting
###############################################################################
# Keras Model Checkpoints (used for early stopping & logging epoch accuracy)
check_point = keras.callbacks.ModelCheckpoint(m.config_model_save_name, monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'min')
early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss', mode = 'min',  patience = 15)
csv_logger = keras.callbacks.CSVLogger(m.config_csv_save_name)


# Define Model Compilation
model.compile(loss='categorical_crossentropy',
              optimizer = Adam(),
              metrics = ['categorical_accuracy'])

model.fit(train_gen,
          epochs = mc_epochs,
          validation_data = valid_gen,
          steps_per_epoch = tsteps,
          validation_steps = vsteps,
          callbacks = [check_point, early_stop, csv_logger, lr_schedule.lr_scheduler()],
          class_weight = class_weight_dict)

train_end_time = time.time()
m.sec_to_time_elapsed(train_end_time, train_start_time)


### Plot Model Progress
###############################################################################
# Accuracy
m.plot_training_progress(csv_file_path = m.config_csv_save_name,
                         train_metric = 'categorical_accuracy',
                         validation_metric = 'val_categorical_accuracy')

# Entropy (Loss)
m.plot_training_progress(csv_file_path = m.config_csv_save_name,
                         train_metric = 'loss',
                         validation_metric = 'val_loss')





### Model Test Set Prediction
###############################################################################
# Predict with Model on Test Set
saved_model = keras.models.load_model(m.config_model_save_name)
pred_values = model.predict(test_x)

# Accuracy on Test Set
true_pos = [int(pred_values[i,np.argmax(test_y[i])] >= 0.5) for i in range(test_y.shape[0])]
true_neg = mf.unnest_list_of_lists([[int(y < 0.5) for i, y in enumerate(pred_values[r,:]) if i != np.argmax(test_y[r])] for r in range(test_y.shape[0])])
true_agg = true_pos + true_neg
pd.DataFrame({'accuracy' : [sum(true_agg) / len(true_agg)],
              'true positive rate' : [sum(true_pos) / len(true_pos)],
              'true negative rate' : [sum(true_neg) / len(true_neg)]})



# Look at Some Predictions
def temp_plot_test_obs(n = 20):
    for i in range(n):
        random_test_obs = random.choice(list(range(test_x.shape[0])))
        class_dict = {0 : 'Chest of drawers', 1 : 'Fireplace', 2 : 'Sofa bed'}
        class_probs = [class_dict.get(i) + ":  " + str(round(x*100,5)) + "%" for i, x in enumerate(pred_values[random_test_obs])]
        actual = class_dict.get(np.argmax(test_y[random_test_obs]))
        plt.imshow(test_x[random_test_obs])
        plt.title("Actual: {a}".format(a = actual) + "\n" + "\n".join(class_probs))
        plt.show()

temp_plot_test_obs(n = 10)