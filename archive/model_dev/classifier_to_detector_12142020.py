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
import skimage
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


### Data Processing: Read Whole Images and Boundign Boxes
###############################################################################
# Television
tv_image_retriever = imm.OpenCVCroppedImageRetriever(class_name = 'Television', max_images = 1000)
tv_img_id_list, tv_coord_list, tv_img_arr = tv_image_retriever.get_whole_images_and_bbox()

# Couch
couch_image_retriever = imm.OpenCVCroppedImageRetriever(class_name = 'Couch', max_images = 1000)
couch_img_id_list, couch_coord_list, couch_img_arr = couch_image_retriever.get_whole_images_and_bbox()


# Coffee Table
ct_image_retriever = imm.OpenCVCroppedImageRetriever(class_name = 'Coffee table', max_images = 1000)
ct_img_id_list, ct_coord_list, ct_img_arr = ct_image_retriever.get_whole_images_and_bbox()




# https://www.analyticsvidhya.com/blog/2018/11/implementation-faster-r-cnn-python-object-detection/?utm_source=blog&utm_medium=build-your-own-object-detection-model-using-tensorflow-api
# https://towardsdatascience.com/creating-your-own-object-detector-ad69dda69c85
# https://www.pyimagesearch.com/2020/06/22/turning-any-cnn-image-classifier-into-an-object-detector-with-keras-tensorflow-and-opencv/



### Data Processing: Create Y Variable
###############################################################################


def extract_resize_image_subset(img_arr, xmin, xmax, ymin, ymax, resize_h, resize_w, decimal = True):
    """
    Retrieve subset of 3d numpy array using decimal coordinates
    (i.e. portion of image with bounding box)
    
    Args:
        img_arr (np.array): 3d numpy array of image
        xmin (float): minimum X-coordinate (expressed as decimal)
        xmax (float): maximum X-coordinate (expressed as decimal)
        ymin (float): minimum Y-coordinate (expressed as decimal)
        ymax (float): maximum Y-coordinate (expressed as decimal)
        decimal (bool): True / False. Indicates whether inputs are decimal positions or integer.
    Returns:
        numpy.array
    """
    if decimal:
        h, w, c = img_arr.shape
        output = img_arr[int(ymin * h):int(ymax * h), int(xmin * w):int(xmax * w)]
        output = resize(output[:,:,:3], (resize_h, resize_w))
    else:
        output = img_arr[ymin:ymax, xmin:xmax]
        output = resize(output[:,:,:3], (resize_h, resize_w))
    return output



unique_image_ids = list(set(ct_img_id_list + couch_img_id_list + tv_img_id_list))

y_classif = []
y_coords = []
x_img = []
x_img_cropped = []

for ID in tqdm.tqdm(unique_image_ids):
    if ID in tv_img_id_list:
        coords = tv_coord_list[tv_img_id_list.index(ID)]
        img_x = tv_img_arr[tv_img_id_list.index(ID)]
        img_x_cropped = extract_resize_image_subset(img_x, coords[0], coords[1], coords[2], coords[3], 200, 200)
        classif = [0, 0, 1]
    elif ID in couch_img_id_list:
        coords = couch_coord_list[couch_img_id_list.index(ID)]
        img_x = couch_img_arr[couch_img_id_list.index(ID)]
        img_x_cropped = extract_resize_image_subset(img_x, coords[0], coords[1], coords[2], coords[3], 200, 200)
        classif = [0, 1, 0]
    else:
        coords = ct_coord_list[ct_img_id_list.index(ID)]
        img_x = ct_img_arr[ct_img_id_list.index(ID)]
        img_x_cropped = extract_resize_image_subset(img_x, coords[0], coords[1], coords[2], coords[3], 200, 200)
        classif = [1, 0, 0]


        
    x_img.append(img_x)
    x_img_cropped.append(img_x_cropped)
    y_coords.append(coords)
    y_classif.append(classif)
    

x_img = np.array(x_img)
x_img_cropped = np.array(x_img_cropped)
y_classif = np.array(y_classif)
y_coords = np.array(y_coords)



### Data Processing: Train, Test, Validation Split
###############################################################################
# Separate Indices for Train, Test, Validation
iters = range(x_img.shape[0])
train_i = random.sample(iters, int(x_img.shape[0] * 0.6))
test_i = random.sample([i for i in iters if i not in train_i], int(x_img.shape[0] * 0.2))
valid_i = [i for i in iters if i not in train_i + test_i]

# Subset Train, Test, Validation
train_x = x_img_cropped[train_i]
test_x = x_img_cropped[test_i]
valid_x = x_img_cropped[valid_i]

train_y_coords = y_coords[train_i]
test_y_coords = y_coords[test_i]
valid_y_coords = y_coords[valid_i]

train_y_classif = y_classif[train_i]
test_y_classif = y_classif[test_i]
valid_y_classif = y_classif[valid_i]


### Data Processing: Class Weights & Augmentation Generator
###############################################################################
# Class Weight Dictionary
class_weight_dict = imm.make_class_weight_dict([np.argmax(x) for x in train_y_classif], return_dict = True)



def rn_id_block(input_x, kernel_size, filters, l2_reg = 0.00001):
    """
    Residual Network Identity Block
        Three-layer block where the shortcut (input) does not have a convolutional layer applied to it
    Args:
        input_x: tensor input
        kernel_size: size of convolutional kernel (integer)
        filters: list of three filter sizes to use in layer block
    Returns:
        tensor
    """
    # Layer 1
    x = Conv2D(filters[0], (1, 1), kernel_regularizer = l2(l2_reg), use_bias = False)(input_x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Layer 2
    x = Conv2D(filters[1], kernel_size, padding='same', kernel_regularizer = l2(l2_reg), use_bias = False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Layer 3
    x = Conv2D(filters[2], (1, 1), kernel_regularizer = l2(l2_reg), use_bias = False)(x)
    x = BatchNormalization()(x)
    
    # Skip Connection
    x = Add()([x, input_x])
    x = Activation('relu')(x)
    return x


def rn_conv_block(input_x, kernel_size, filters, strides=(2, 2), l2_reg = 0.00001):
    """
    Residual Network Convolutional Block
        Three-layer block where the shortcut (input) has a convolutional layer applied to it
    Args:
        input_x: tensor input
        kernel_size: size of convolutional kernel (integer)
        filters: list of three filter sizes to use in layer block
    Returns:
        tensor
    """
    # Layer 1
    x = Conv2D(filters[0], (1, 1), strides=strides, kernel_regularizer = l2(l2_reg), use_bias = False)(input_x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Layer 2
    x = Conv2D(filters[1], kernel_size, padding='same', kernel_regularizer = l2(l2_reg), use_bias = False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Layer 3
    x = Conv2D(filters[2], (1, 1), kernel_regularizer = l2(l2_reg), use_bias = False)(x)
    x = BatchNormalization()(x)

    # Skip Connection
    shortcut = Conv2D(filters[2], (1, 1), strides=strides, use_bias = False)(input_x)
    shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x



def resnet_conv_n_layer(kernel_size = (3,3), img_height = 200,
                         img_width = 200, color_mode = 'rgb', 
                         activ = 'relu', n_classes = 3):
    # Input shape
    if color_mode == "rgb":
        config_input_shape = (img_height, img_width, 3)
    else:
        config_input_shape = (img_height, img_width, 1)
        
    x_input = Input(config_input_shape)

    # Layer 1
    x = ZeroPadding2D(padding=(3, 3))(x_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', use_bias = False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # Block 1
    x = rn_conv_block(x, 3, [64, 64, 256], strides=(1, 1))
    x = rn_id_block(x, 3, [64, 64, 256])

    # Block 2
    x = rn_conv_block(x, 3, [128, 128, 512])
    x = rn_id_block(x, 3, [128, 128, 512])

    # Block 3
    x = rn_conv_block(x, 3, [256, 256, 1024])
    x = rn_id_block(x, 3, [256, 256, 1024])
    x = rn_id_block(x, 3, [256, 256, 1024])

    # Block 4
    x = rn_conv_block(x, 3, [512, 512, 2048])
    x = rn_id_block(x, 3, [512, 512, 2048])

    # Dimension Reduction
    x = GlobalAvgPool2D(name = 'global_avg_pooling')(x)
    x = Flatten()(x)
    x = Dense(n_classes, activation='softmax')(x)
    
    model = Model(inputs = x_input, outputs = x, name = 'resnet_31_layer') 
    return model








class RandomImageAugmenter:
    def __init__(self, image_array,
                 max_rotation = 50,
                 min_rotation = -50):
        self.image_array = image_array
        self.max_rotation = max_rotation
        self.min_rotation = min_rotation
    
    def rotate_randomly(self, img):
        return skimage.transform.rotate(img, random.uniform(self.max_rotation, self.min_rotation))
    
    def add_random_noise(self, img):
        return skimage.util.random_noise(img, mode = 'gaussian')
    
    def flip_randomly(self, img):
        function_dict = {0 : img,
                         1 : np.flipud(img),
                         2 : np.fliplr(img),
                         3 : np.flipud(np.fliplr(img))}
        rand_int = random.choice([0, 1, 2, 3])
        return function_dict.get(rand_int)
    
    def augment(self):
        norm_img = self.rotate_randomly(self.image_array)
        norm_img = self.flip_randomly(norm_img)
        #if random.choice([0,1]) == 1:
        #    norm_img = self.add_random_noise(norm_img)
        return norm_img
        



def augmented_batch_generator(x_arr, y_arr, batch_size = 20):
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
                    x_arr_aug = np.array([RandomImageAugmenter(x).augment() for x in x_arr[batch_list]])
                    yield x_arr_aug, y_arr[batch_list]
                    batch_list=[]




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
                    x_arr_aug = np.array([np.fliplr(x) if (j % h_flip_every) == 0 else x for j, x in enumerate(x_arr[batch_list])])
                    x_arr_aug = np.array([np.flipud(x) if (j % v_flip_every) == 0 else x for j, x in enumerate(x_arr_aug)])
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
mc_batch_size = 40
mc_epochs = 50
mc_learning_rate = 0.001
mc_dropout = 0.2

# Calculate Training Steps
tsteps = int(train_x.shape[0]) // mc_batch_size
vsteps = int(valid_x.shape[0]) // mc_batch_size

# Create Learning Rate Schedule
lr_schedule = m.CyclicalRateSchedule(min_lr = 0.000015,
                                     max_lr = 0.0001,
                                     n_epochs = 200,
                                     warmup_epochs = 10,
                                     cooldown_epochs = 1,
                                     cycle_length = 6,
                                     logarithmic = True,
                                     decrease_factor = 0.9)

lr_schedule.plot_cycle()

# Create Augmentation Generator Objects
train_gen = augmented_batch_generator(train_x, train_y_classif, batch_size = mc_batch_size)
valid_gen = augmented_batch_generator(valid_x, valid_y_classif, batch_size = mc_batch_size)
test_gen = augmented_batch_generator(test_x, test_y_classif, batch_size = mc_batch_size)

### Model Architecture
###############################################################################
# Clear Session (this removes any trained model from your PC's memory)
train_start_time = time.time()
keras.backend.clear_session()

# 19-Layer CNN
model = cnn_19_layer(n_classes = 3, kernel_size = 3)

#model = m.resnet_conv_50_layer(img_height = 200, img_width = 200, n_classes = 3)

### Model Fitting
###############################################################################
# Keras Model Checkpoints (used for early stopping & logging epoch accuracy)
check_point = keras.callbacks.ModelCheckpoint(m.config_model_save_name, monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'min')
early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss', mode = 'min',  patience = 15)
csv_logger = keras.callbacks.CSVLogger(m.config_csv_save_name)


# Define Model Compilation
model.compile(loss='categorical_crossentropy',
              optimizer = Adam(),
              metrics = ['binary_accuracy'])

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
                         train_metric = 'binary_accuracy',
                         validation_metric = 'val_binary_accuracy')

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
true_pos = [int(pred_values[i,np.argmax(test_y_classif[i])] >= 0.5) for i in range(test_y_classif.shape[0])]
true_neg = mf.unnest_list_of_lists([[int(y < 0.5) for i, y in enumerate(pred_values[r,:]) if i != np.argmax(test_y_classif[r])] for r in range(test_y_classif.shape[0])])
true_agg = true_pos + true_neg
pd.DataFrame({'accuracy' : [sum(true_agg) / len(true_agg)],
              'true positive rate' : [sum(true_pos) / len(true_pos)],
              'true negative rate' : [sum(true_neg) / len(true_neg)]})



# Look at Some Predictions
def temp_plot_test_obs(n = 20):
    for i in range(n):
        random_test_obs = random.choice(list(range(test_x.shape[0])))
        print(random_test_obs)
        class_dict = {0 : 'Coffee table', 1 : 'Couch', 2 : 'Television'}
        class_probs = [class_dict.get(i) + ":  " + str(round(x*100,5)) + "%" for i, x in enumerate(pred_values[random_test_obs])]
        actual = class_dict.get(np.argmax(test_y_classif[random_test_obs]))
        plt.imshow(test_x[random_test_obs])
        plt.title("Actual: {a}".format(a = actual) + "\n" + "\n".join(class_probs))
        plt.show()

temp_plot_test_obs(n = 1)





### Localize
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################


plt.imshow(x_img[100])



def get_sliding_boxes(img_arr, h = 20, w = 40):
    coords = []
    for r in range(0, img_arr.shape[0], w):
        for c in range(0, img_arr.shape[1], h):
            coords.append((r, r + w, c, c + h))
    return coords


def get_sliding_box_predictions(model, img_arr, box_h, box_w, img_w, img_h):
    box_coords = get_sliding_boxes(img_arr, h = box_h, w = box_w)
    pred = []
    for bc in tqdm.tqdm(box_coords):
        bc_img = extract_resize_image_subset(img_arr, bc[0], bc[1], bc[2], bc[3], img_w, img_h, decimal = False)
        print(bc_img.shape)
        pred.append(model.predict(np.expand_dims(bc_img, axis = 0))[0])
    return box_coords, pred



box_coords, pred = get_sliding_box_predictions(model = saved_model,
                                               img_arr = x_img[100],
                                               box_h = 50,
                                               box_w = 50,
                                               img_w = 200,
                                               img_h = 200)





def get_sliding_box_predictions(model, img_arr, box_h, box_w, img_w, img_h, n_classes):
    box_coords = get_sliding_boxes(img_arr, h = box_h, w = box_w)
    empty_arr = np.zeros((img_w, img_h, n_classes))
    pred = []
    for bc in tqdm.tqdm(box_coords):
        bc_img = extract_resize_image_subset(img_arr, bc[0], bc[1], bc[2], bc[3], img_w, img_h, decimal = False)
        pred = model.predict(np.expand_dims(bc_img, axis = 0))[0]
        for s in range(pred.shape[0]):
            empty_arr[bc[0]:bc[1], bc[2]:bc[3], s] = pred[s]
    return empty_arr
    

pred_arr = get_sliding_box_predictions(model = saved_model,
                                               img_arr = x_img[100],
                                               box_h = 50,
                                               box_w = 50,
                                               img_w = 200,
                                               img_h = 200,
                                               n_classes = 3)


plt.imshow(pred_arr)

  
box_coords, pred = get_sliding_box_predictions(model = saved_model,
                                               img_arr = x_img[100],
                                               box_h = 20,
                                               box_w = 40,
                                               img_w = 200,
                                               img_h = 200,
                                               n_classes = 3)

temp_bbc = get_sliding_boxes(x_img[0])[10]

temp = extract_resize_image_subset(x_img[0], temp_bbc[0], temp_bbc[1], temp_bbc[2], temp_bbc[3], 200, 200, decimal = False)


plt.imshow(temp)


plt.imshow(temp)


plt.imshow(x_img[0])
















