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
import src.config_data_processing as cdp
import src.image_manipulation as imm
import src.misc_functions as mf
import src.modeling as m


### Retrieve & Process Class Data
###############################################################################   
class_processor = imm.OpenCVMultiClassProcessor(class_list = ['Piano', 'Computer monitor', 'Kitchen & dining room table'], max_images = 3000)


proc_data_dict = class_processor.get_train_test_valid_data()
train_x = proc_data_dict.get('TRAIN X')
test_x = proc_data_dict.get('TEST X')
valid_x = proc_data_dict.get('VALIDATION X')
train_y = proc_data_dict.get('TRAIN Y')
test_y = proc_data_dict.get('TEST Y')
valid_y = proc_data_dict.get('VALIDATION Y')
class_weight_dict = proc_data_dict.get('CLASS WEIGHT DICT')


### Remove Arrays with All Zeroes (black images... will break neural net)
###############################################################################
# Training Set
nan = [i for i, x in enumerate(train_x) if np.isnan(np.sum(x))]
inf = [i for i, x in enumerate(train_x) if math.isinf(np.sum(x))]
near_zero = [i for i, x in enumerate(train_x) if (np.sum(x == 0) / np.sum(x != 0)) > 5]
remove = list(set(nan + inf + near_zero))
keep = [i for i in range(train_x.shape[0]) if i not in remove]
train_x = train_x[keep]
train_y = train_y[keep]


# Validation Set
nan = [i for i, x in enumerate(valid_x) if np.isnan(np.sum(x))]
inf = [i for i, x in enumerate(valid_x) if math.isinf(np.sum(x))]
near_zero = [i for i, x in enumerate(valid_x) if (np.sum(x == 0) / np.sum(x != 0)) > 5]
remove = list(set(nan + inf + near_zero))
keep = [i for i in range(valid_x.shape[0]) if i not in remove]
valid_x = valid_x[keep]
valid_y = valid_y[keep]

# Test Set
nan = [i for i, x in enumerate(test_x) if np.isnan(np.sum(x))]
inf = [i for i, x in enumerate(test_x) if math.isinf(np.sum(x))]
near_zero = [i for i, x in enumerate(test_x) if (np.sum(x == 0) / np.sum(x != 0)) > 5]
remove = list(set(nan + inf + near_zero))
keep = [i for i in range(test_x.shape[0]) if i not in remove]
test_x = test_x[keep]
test_y = test_y[keep]


### One-Hot Encode Y-Values
###############################################################################
train_y = np.array([[1 if t == i else 0 for i, x in enumerate(np.unique(train_y))] for t in train_y])
test_y = np.array([[1 if t == i else 0 for i, x in enumerate(np.unique(test_y))] for t in test_y])
valid_y = np.array([[1 if t == i else 0 for i, x in enumerate(np.unique(valid_y))] for t in valid_y])


### Define Keras Configuration
###############################################################################

# Define Number of Steps Per Epoch (observations / batch size)
tsteps = int(train_x.shape[0]) // m.config_batch_size
vsteps = int(valid_x.shape[0]) // m.config_batch_size

# Create Learning Rate Schedule
lr_schedule = m.CyclicalRateSchedule(min_lr = 1e-05,
                                     max_lr = 0.0005,
                                     n_epochs = 25,
                                     warmup_epochs = 1,
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




