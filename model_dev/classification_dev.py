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


### Retrieve & Process Class Data
###############################################################################   
class_processor = imm.OpenCVMultiClassProcessor(class_list = ['Bed', 'Sink', 'Television'], max_images = 2000)


proc_data_dict = class_processor.get_train_test_valid_data()

# Full Image Arrays
train_x = proc_data_dict.get('TRAIN X')
test_x = proc_data_dict.get('TEST X')
valid_x = proc_data_dict.get('VALIDATION X')

# Image Arrays Cropped to Objects
train_obj_x = proc_data_dict.get('TRAIN OBJECT X')
test_obj_x = proc_data_dict.get('TEST OBJECT X')
valid_obj_x = proc_data_dict.get('VALIDATION OBJECT X')

# Bounding Boxes
train_bbox = proc_data_dict.get('TRAIN BBOX')
test_bbox = proc_data_dict.get('TEST BBOX')
valid_bbox = proc_data_dict.get('VALIDATION BBOX')

# Response Variable & Class Weights
train_y = proc_data_dict.get('TRAIN Y')
test_y = proc_data_dict.get('TEST Y')
valid_y = proc_data_dict.get('VALIDATION Y')
class_weight_dict = proc_data_dict.get('CLASS WEIGHT DICT')



### Augment Training Data
###############################################################################   
# Flip Half of Arrays Left/Right and Half Up/Down
train_aug_x = np.array([np.fliplr(x) if j%2 == 0 else np.flipud(x) for j, x in enumerate(train_x)])

# Look at Difference: Left/Right
plt.imshow(train_x[0])
plt.imshow(train_aug_x[0])

# Look at Difference: Up/Down
plt.imshow(train_x[1])
plt.imshow(train_aug_x[1])

# Combine
train_x = np.vstack((train_x, train_aug_x))
train_y = np.vstack((train_y, train_y))

### Define Keras Configuration
###############################################################################

# Define Number of Steps Per Epoch (observations / batch size)
tsteps = int(train_x.shape[0]) // m.config_batch_size
vsteps = int(valid_x.shape[0]) // m.config_batch_size

# Create Learning Rate Schedule
lr_schedule = m.CyclicalRateSchedule(min_lr = 0.00002,
                                     max_lr = 0.00015,
                                     n_epochs = 25,
                                     warmup_epochs = 3,
                                     cooldown_epochs = 1,
                                     cycle_length = 5,
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
              optimizer = Adam(),
              metrics = ['categorical_accuracy'])

model.fit(train_x, train_y,
          epochs = 25,
          validation_data = (valid_x, valid_y),
          steps_per_epoch = tsteps,
          callbacks = [check_point, early_stop, lr_schedule.lr_scheduler()],
          class_weight = class_weight_dict)

train_end_time = time.time()
m.sec_to_time_elapsed(train_end_time, train_start_time)


### Model Test Set Prediction
###############################################################################
# Load Model from Best Epoch
saved_model = keras.models.load_model(m.config_model_save_name)
pred_values = saved_model.predict(test_x)

# Accuracy on Test Set
true_pos = [int(pred_values[i,np.argmax(test_y[i])] >= 0.5) for i in range(test_y.shape[0])]
true_neg = mf.unnest_list_of_lists([[int(y < 0.5) for i, y in enumerate(pred_values[r,:]) if i != np.argmax(test_y[r])] for r in range(test_y.shape[0])])
true_agg = true_pos + true_neg
tpr = sum(true_pos) / len(true_pos)
tnr = sum(true_neg) / len(true_neg)
acc = sum(true_agg) / len(true_agg)
pd.DataFrame({'accuracy' : [acc], 'true positive rate' : [tpr], 'true negative rate' : [tnr]})



# Look at Some Predictions
def temp_plot_test_obs(n = 20):
    for i in range(n):
        random_test_obs = random.choice(list(range(test_x.shape[0])))
        class_dict = {0 : 'Kitchen & dining room table', 1 : "Piano", 2 : "Computer monitor"}
        class_probs = [class_dict.get(i) + ":  " + str(round(x*100,5)) + "%" for i, x in enumerate(pred_values[random_test_obs])]
        actual = class_dict.get(np.argmax(test_y[random_test_obs]))
        plt.imshow(test_x[random_test_obs])
        plt.title("Actual: {a}".format(a = actual) + "\n" + "\n".join(class_probs))
        plt.show()

temp_plot_test_obs()




