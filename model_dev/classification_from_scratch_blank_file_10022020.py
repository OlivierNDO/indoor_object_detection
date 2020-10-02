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

# Cropped 'Fireplace' Images


# Cropped 'Sofa bed' Images



### Data Processing: Create Y Variable
###############################################################################
# to be consistent, we should always order classes alphabetically

# Create and Combine Binary Variables

# Delete Objects from Memory
del x1; del x2; del x3; del binary1; del binary2; del binary3;


### Data Processing: Train, Test, Validation Split
###############################################################################
# Split X and Y into 70/30 Train/Test

# Split Test Set into 15/15 Test/Validation


### Data Processing: Class Weights
###############################################################################

# Using Function
class_weight_dict = imm.make_class_weight_dict([np.argmax(x) for x in train_y], return_dict = True)


### Model Configuration
###############################################################################
# Parameters
mc_batch_size = 20
mc_epochs = 2
mc_learning_rate = 0.001
mc_dropout = 0.2

# Calculate Training Steps


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


# Convolutional Layer 2


# Convolutional Layer 3



# Convolutional Layer 4



# Convolutional Layer 5 (same as 4)


# Convolutional Layer 6


# Convolutional Layer 7


# Dense Layers (output size 3 equal to number of classes)


# Create Model Object
model = Model(inputs = x_input, outputs = x, name = 'cnn_from_scratch') 
model.summary()

### Model Fitting
###############################################################################
# Keras Model Checkpoints (used for early stopping & logging epoch accuracy)
check_point = keras.callbacks.ModelCheckpoint(m.config_model_save_name, monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'min')
early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss', mode = 'min',  patience = m.config_max_worse_epochs)
csv_logger = keras.callbacks.CSVLogger(m.config_csv_save_name)


# Define Model Compilation
model.compile(loss='categorical_crossentropy',
              optimizer = Adam(learning_rate = mc_learning_rate),
              metrics = ['categorical_accuracy'])

model.fit(train_x, train_y,
          epochs = mc_epochs,
          validation_data = (valid_x, valid_y),
          steps_per_epoch = tsteps,
          validation_steps = vsteps,
          callbacks = [check_point, early_stop, csv_logger],
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