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

# Path to Images (3500 images in OneDrive)
local_image_folder = 'C:/local_images/'

# Path to Text File
dict_write_folder = f'{my_project_folder}data/processed_data/'
dict_list_save_name = 'object_dict_list.pkl'

# Path to Transfer Learning Weights
weights_path = f'{my_project_folder}weights/yolo-voc.weights'


### Configuration - Model and Data Processing
###############################################################################
generator_config = {
    'IMAGE_H' : 416, 
    'IMAGE_W' : 416,
    'GRID_H' : 13,  
    'GRID_W' : 13,
    'BOX' : 5,
    'LABELS' : ['Television', 'Couch', 'Coffee table', 'Piano'],
    'CLASS' : len(['Television', 'Couch', 'Coffee table', 'Piano']),
    'ANCHORS' : [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828],
    'BATCH_SIZE' : 4,
    'TRUE_BOX_BUFFER' : 50,
    'LAMBDA_NO_OBJECT' : 1.0,
    'LAMBDA_OBJECT':  5.0,
    'LAMBDA_COORD' : 1.0,
    'LAMBDA_CLASS' : 1.0
}



### Define Functions
###############################################################################
class ImageReader(object):
    def __init__(self,
                 IMAGE_H = generator_config['IMAGE_H'],
                 IMAGE_W = generator_config['IMAGE_W']):
        """
        Read & resize images stored in files and return bounding box
        information given a dictionary with the following format:
            {'filename': 'outer_folder/subfolder/image_name.jpg',
             'height':   333,
             'width':    500,
             'object': [{'name': 'bird', 'xmax': 318, 'xmin': 284, 'ymax': 184, 'ymin': 100},
                        {'name': 'bird',  'xmax': 198, 'xmin': 112, 'ymax': 209, 'ymin': 146}]}
        Args:
            IMAGE_H (int): image height
            IMAGE_W (int): image width
        """
        self.IMAGE_H = IMAGE_H
        self.IMAGE_W = IMAGE_W
    
    def fit(self,train_instance):
        if not isinstance(train_instance,dict):
            train_instance = {'filename':train_instance}
        image_name = train_instance['filename']
        image = np.asarray(load_img(image_name))
        h, w, c = image.shape
        image = image[:,:,::-1]
        image = resize(image, (self.IMAGE_H, self.IMAGE_W))
            
        if 'object' in train_instance.keys():
            all_objs = copy.deepcopy(train_instance['object'])     
            for obj in all_objs:
                for attr in ['xmin', 'xmax']:
                    obj[attr] = int(obj[attr] * float(self.IMAGE_W) / w)
                    obj[attr] = max(min(obj[attr], self.IMAGE_W), 0)

                for attr in ['ymin', 'ymax']:
                    obj[attr] = int(obj[attr] * float(self.IMAGE_H) / h)
                    obj[attr] = max(min(obj[attr], self.IMAGE_H), 0)
        else:
            return image
        return image, all_objs


def normalize(image):
    """
    Normalize numpy image array by dividing floats by 255.
    Args:
        image (np.array): 3d numpy image array
    """
    return image / 255.


class BestAnchorBoxFinder(object):
    def __init__(self, ANCHORS = generator_config.get('ANCHORS')):
        """
        Given a fixed set of initial anchors, find anchor boxes that maximize
        intersection over union (IOU)
        Args:
            ANCHORS (np.array): array of floats (must be even number length)
        """
        self.anchors = [BoundBox(0, 0, ANCHORS[2*i], ANCHORS[2*i+1]) 
                        for i in range(int(len(ANCHORS)//2))]
        
    def _interval_overlap(self,interval_a, interval_b):
        x1, x2 = interval_a
        x3, x4 = interval_b
        if x3 < x1:
            if x4 < x1:
                return 0
            else:
                return min(x2,x4) - x1
        else:
            if x2 < x3:
                 return 0
            else:
                return min(x2,x4) - x3  

    def bbox_iou(self,box1, box2):
        intersect_w = self._interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
        intersect_h = self._interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])  
        intersect = intersect_w * intersect_h
        w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
        w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
        union = w1*h1 + w2*h2 - intersect
        return float(intersect) / union
    
    def find(self,center_w, center_h):
        best_anchor = -1
        max_iou     = -1
        shifted_box = BoundBox(0, 0,center_w, center_h)
        for i in range(len(self.anchors)):
            anchor = self.anchors[i]
            iou    = self.bbox_iou(shifted_box, anchor)
            if max_iou < iou:
                best_anchor = i
                max_iou     = iou
        return(best_anchor,max_iou)
    
    
class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, confidence=None,classes=None):
        """
        Returns labels and confidence score given a list of classes
        and vector of predicted probabilities
        Args:
            xmin (float): minimum x-axis value in bounding box
            ymin (float): minimum y-axis value in bounding box
            xmax (float): maximum x-axis value in bounding box
            ymax (float): maximum y-axis value in bounding box
        
        """
        self.xmin, self.ymin = xmin, ymin
        self.xmax, self.ymax = xmax, ymax
        self.confidence      = confidence
        self.set_class(classes)
        
    def set_class(self,classes):
        self.classes = classes
        self.label   = np.argmax(self.classes) 
        
    def get_label(self):  
        return(self.label)
    
    def get_score(self):
        return(self.classes[self.label])
    
    
def rescale_centerxy(obj, config = generator_config):
    """
    Find centermost point within a grid overlaid on numpy array
    Args:
        obj (dict): dictionary with keys 'xmin', 'xmax', 'ymin', 'ymax'
        config (dict): dictionary with keys 'IMAGE_W', 'GRID_W', 'IMAGE_H' and 'GRID_H'
    Returns:
        tuple
    """
    center_x = .5 * (obj['xmin'] + obj['xmax'])
    center_x = center_x / (float(config['IMAGE_W']) / config['GRID_W'])
    center_y = .5 * (obj['ymin'] + obj['ymax'])
    center_y = center_y / (float(config['IMAGE_H']) / config['GRID_H'])
    return (center_x, center_y)


def rescale_centerwh(obj, config = generator_config):
    """
    Find centermost point within a grid overlaid on numpy array
    Args:
        obj (dict): dictionary with keys 'xmin', 'xmax', 'ymin', 'ymax'
        config (dict): dictionary with keys 'IMAGE_W', 'GRID_W', 'IMAGE_H' and 'GRID_H'
    Returns:
        tuple
    """
    center_w = (obj['xmax'] - obj['xmin']) / (float(config['IMAGE_W']) / config['GRID_W']) 
    center_h = (obj['ymax'] - obj['ymin']) / (float(config['IMAGE_H']) / config['GRID_H']) 
    return(center_w, center_h)


class SimpleBatchGenerator(Sequence):
    def __init__(self, images, config = generator_config, shuffle = True):
        """
        Batch generator class to feed minibatches into Keras Model() object
        Args:
            config : dictionary with keys:
                'IMAGE_H', 'IMAGE_W', 'GRID_H', 'GRID_W', 'LABELS',
                'ANCHORS', 'BATCH_SIZE', 'TRUE_BOX_BUFFER'
        """
        self.config = config
        self.config["BOX"] = int(len(self.config['ANCHORS'])/2)
        self.config["CLASS"] = len(self.config['LABELS'])
        self.images = images
        self.bestAnchorBoxFinder = BestAnchorBoxFinder(config['ANCHORS'])
        self.imageReader = ImageReader(config['IMAGE_H'],config['IMAGE_W'])
        self.shuffle = shuffle
        if self.shuffle: 
            np.random.shuffle(self.images)
            
    def __len__(self):
        return int(np.ceil(float(len(self.images))/self.config['BATCH_SIZE']))  
    
    def __getitem__(self, idx):
        '''
        == input == 
        
        idx : non-negative integer value e.g., 0
        
        == output ==
        
        x_batch: The numpy array of shape  (BATCH_SIZE, IMAGE_H, IMAGE_W, N channels).
            
            x_batch[iframe,:,:,:] contains a iframeth frame of size  (IMAGE_H,IMAGE_W).
            
        y_batch:

            The numpy array of shape  (BATCH_SIZE, GRID_H, GRID_W, BOX, 4 + 1 + N classes). 
            BOX = The number of anchor boxes.

            y_batch[iframe,igrid_h,igrid_w,ianchor,:4] contains (center_x,center_y,center_w,center_h) 
            of ianchorth anchor at  grid cell=(igrid_h,igrid_w) if the object exists in 
            this (grid cell, anchor) pair, else they simply contain 0.

            y_batch[iframe,igrid_h,igrid_w,ianchor,4] contains 1 if the object exists in this 
            (grid cell, anchor) pair, else it contains 0.

            y_batch[iframe,igrid_h,igrid_w,ianchor,5 + iclass] contains 1 if the iclass^th 
            class object exists in this (grid cell, anchor) pair, else it contains 0.


        b_batch:

            The numpy array of shape (BATCH_SIZE, 1, 1, 1, TRUE_BOX_BUFFER, 4).

            b_batch[iframe,1,1,1,ibuffer,ianchor,:] contains ibufferth object's 
            (center_x,center_y,center_w,center_h) in iframeth frame.

            If ibuffer > N objects in iframeth frame, then the values are simply 0.

            TRUE_BOX_BUFFER has to be some large number, so that the frame with the 
            biggest number of objects can also record all objects.

            The order of the objects do not matter.

            This is just a hack to easily calculate loss. 
        
        '''
        l_bound = idx*self.config['BATCH_SIZE']
        r_bound = (idx+1)*self.config['BATCH_SIZE']

        if r_bound > len(self.images):
            r_bound = len(self.images)
            l_bound = r_bound - self.config['BATCH_SIZE']

        instance_count = 0
        
        ## prepare empty storage space: this will be output
        x_batch = np.zeros((r_bound - l_bound, self.config['IMAGE_H'], self.config['IMAGE_W'], 3))                         # input images
        b_batch = np.zeros((r_bound - l_bound, 1     , 1     , 1    ,  self.config['TRUE_BOX_BUFFER'], 4))   # list of self.config['TRUE_self.config['BOX']_BUFFER'] GT boxes
        y_batch = np.zeros((r_bound - l_bound, self.config['GRID_H'],  self.config['GRID_W'], self.config['BOX'], 4+1+len(self.config['LABELS'])))                # desired network output

        for train_instance in self.images[l_bound:r_bound]:
            # augment input image and fix object's position and size
            img, all_objs = self.imageReader.fit(train_instance)
            
            # construct output from object's x, y, w, h
            true_box_index = 0
            
            for obj in all_objs:
                if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin'] and obj['name'] in self.config['LABELS']:
                    center_x, center_y = rescale_centerxy(obj,self.config)
                    
                    grid_x = int(np.floor(center_x))
                    grid_y = int(np.floor(center_y))

                    if grid_x < self.config['GRID_W'] and grid_y < self.config['GRID_H']:
                        obj_indx  = self.config['LABELS'].index(obj['name'])
                        center_w, center_h = rescale_centerwh(obj,self.config)
                        box = [center_x, center_y, center_w, center_h]
                        best_anchor,max_iou = self.bestAnchorBoxFinder.find(center_w, center_h)
                                
                        # assign ground truth x, y, w, h, confidence and class probs to y_batch
                        # it could happen that the same grid cell contain 2 similar shape objects
                        # as a result the same anchor box is selected as the best anchor box by the multiple objects
                        # in such ase, the object is over written
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 0:4] = box # center_x, center_y, w, h
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 4  ] = 1. # ground truth confidence is 1
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 5+obj_indx] = 1 # class probability of the object
                        
                        # assign the true box to b_batch
                        b_batch[instance_count, 0, 0, 0, true_box_index] = box
                        
                        true_box_index += 1
                        true_box_index = true_box_index % self.config['TRUE_BOX_BUFFER']
                            
            x_batch[instance_count] = img
            # increase instance counter in current batch
            instance_count += 1  
        return [x_batch, b_batch], y_batch

    def on_epoch_end(self):
        if self.shuffle: 
            np.random.shuffle(self.images)


def space_to_depth_x2(x):
    """
    Function to implement the orgnization layer (thanks to github.com/allanzelener/YAD2K)
    """
    return tf.nn.space_to_depth(x, block_size=2)
            


class WeightReader:
    def __init__(self, weight_file):
        self.offset = 4
        self.all_weights = np.fromfile(weight_file, dtype='float32')
        
    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset-size:self.offset]
    
    def reset(self):
        self.offset = 4




    
    
    


### Define Model Architecture
###############################################################################
def conv_batchnorm_leaky_layer(x, neurons, kernel = (3,3), strides = (1,1),
                               padding = 'same', use_bias = False, use_maxpooling = False, pool_size = (2,2),
                               conv_name = None, norm_name = None):
    if conv_name is None:
        xx = Conv2D(neurons, kernel, strides = strides, padding = padding, use_bias=False)(x)
    else:
        xx = Conv2D(neurons, kernel, strides = strides, padding = padding, use_bias=False, name = conv_name)(x)
    if norm_name is None:
        xx = BatchNormalization()(xx)
    else:
        xx = BatchNormalization(name = norm_name)(xx)
    xx = LeakyReLU(alpha=0.1)(xx)
    if use_maxpooling:
        xx = MaxPooling2D(pool_size = pool_size)(xx)
    return xx


def network_in_network(input_tensor, dims):
    for d in dims:
        input_tensor = conv_batchnorm_leaky_layer(x = input_tensor, neurons = d[0], kernel = (d[1], d[1]))
    return input_tensor


def reorg(input_tensor, stride):
    _, h, w, c = input_tensor.get_shape().as_list() 

    channel_first = Permute((3, 1, 2))(input_tensor)
    
    reshape_tensor = Reshape((c // (stride ** 2), h, stride, w, stride))(channel_first)
    permute_tensor = Permute((3, 5, 1, 2, 4))(reshape_tensor)
    target_tensor = Reshape((-1, h // stride, w // stride))(permute_tensor)
    
    channel_last = Permute((2, 3, 1))(target_tensor)
    return Reshape((h // stride, w // stride, -1))(channel_last)



def yolo_v2_convnet(IMAGE_H = generator_config.get('IMAGE_H'),
                    IMAGE_W = generator_config.get('IMAGE_W'),
                    CLASS = generator_config.get('CLASS'),
                    TRUE_BOX_BUFFER = generator_config.get('TRUE_BOX_BUFFER'),
                    BOX = generator_config.get('BOX'),
                    GRID_H = generator_config.get('GRID_H'),
                    GRID_W = generator_config.get('GRID_W'),
                    n_channels = 3,
                    pool_size = (2,2)):
    input_image = Input(shape=(IMAGE_H, IMAGE_W, 3))
    true_boxes  = Input(shape=(1, 1, 1, TRUE_BOX_BUFFER, 4))
    
    x = conv_batchnorm_leaky_layer(x = input_image, neurons = 32, use_maxpooling = True, conv_name = 'conv_1', norm_name = 'norm_1')
    x = conv_batchnorm_leaky_layer(x = x, neurons = 64, use_maxpooling = True, conv_name = 'conv_2', norm_name = 'norm_2')
    x = conv_batchnorm_leaky_layer(x = x, neurons = 128, conv_name = 'conv_3', norm_name = 'norm_3')
    x = conv_batchnorm_leaky_layer(x = x, neurons = 64, kernel = (1,1), conv_name = 'conv_4', norm_name = 'norm_4')
    x = conv_batchnorm_leaky_layer(x = x, neurons = 128, use_maxpooling = True, conv_name = 'conv_5', norm_name = 'norm_5')
    
    x = conv_batchnorm_leaky_layer(x = x, neurons = 256, conv_name = 'conv_6', norm_name = 'norm_6')
    x = conv_batchnorm_leaky_layer(x = x, neurons = 128, conv_name = 'conv_7', norm_name = 'norm_7')
    x = conv_batchnorm_leaky_layer(x = x, neurons = 256, use_maxpooling = True, conv_name = 'conv_8', norm_name = 'norm_8')
    x = conv_batchnorm_leaky_layer(x = x, neurons = 512, conv_name = 'conv_9', norm_name = 'norm_9')
    x = conv_batchnorm_leaky_layer(x = x, neurons = 256, kernel = (1,1), conv_name = 'conv_10', norm_name = 'norm_10')
    
    x = conv_batchnorm_leaky_layer(x = x, neurons = 512, conv_name = 'conv_11', norm_name = 'norm_11')
    x = conv_batchnorm_leaky_layer(x = x, neurons = 256, kernel = (1,1), conv_name = 'conv_12', norm_name = 'norm_12')
    x = conv_batchnorm_leaky_layer(x = x, neurons = 512, conv_name = 'conv_13', norm_name = 'norm_13')
    skip_connection = x
    x = MaxPooling2D(pool_size = pool_size)(x)
    x = conv_batchnorm_leaky_layer(x = x, neurons = 1024, conv_name = 'conv_14', norm_name = 'norm_14')
    x = conv_batchnorm_leaky_layer(x = x, neurons = 512, kernel = (1,1), conv_name = 'conv_15', norm_name = 'norm_15')
    x = conv_batchnorm_leaky_layer(x = x, neurons = 1024, conv_name = 'conv_16', norm_name = 'norm_16')
    x = conv_batchnorm_leaky_layer(x = x, neurons = 512, kernel = (1,1), conv_name = 'conv_17', norm_name = 'norm_17')
    x = conv_batchnorm_leaky_layer(x = x, neurons = 1024, conv_name = 'conv_18', norm_name = 'norm_18')
    x = conv_batchnorm_leaky_layer(x = x, neurons = 1024, conv_name = 'conv_19', norm_name = 'norm_19')
    x = conv_batchnorm_leaky_layer(x = x, neurons = 1024, conv_name = 'conv_20', norm_name = 'norm_20')
    
    skip_connection = conv_batchnorm_leaky_layer(x = skip_connection, neurons = 1024, kernel = (1,1), conv_name = 'conv_21', norm_name = 'norm_21')
    skip_connection = Lambda(space_to_depth_x2)(skip_connection)
    x = concatenate([skip_connection, x])
    x = conv_batchnorm_leaky_layer(x = x, neurons = 1024, conv_name = 'conv_22', norm_name = 'norm_22')
    x = Conv2D(BOX * (4 + 1 + CLASS), (1,1), strides=(1,1), padding='same', name='conv_23')(x)
    output = Reshape((GRID_H, GRID_W, BOX, 4 + 1 + CLASS))(x)
    output = Lambda(lambda args: args[0])([output, true_boxes])
    model = Model([input_image, true_boxes], output)
    return model


def yolo_noskip_convnet(IMAGE_H = generator_config.get('IMAGE_H'),
                    IMAGE_W = generator_config.get('IMAGE_W'),
                    TRUE_BOX_BUFFER = generator_config.get('TRUE_BOX_BUFFER'),
                    GRID_H = generator_config.get('GRID_H'),
                    GRID_W = generator_config.get('GRID_W'),
                    BOX = generator_config.get('BOX'),
                    CLASS = generator_config.get('CLASS'),
                    n_channels = 3,
                    pool_size = (2,2)):
    input_image = Input(shape=(IMAGE_H, IMAGE_W, 3))
    true_boxes  = Input(shape=(1, 1, 1, TRUE_BOX_BUFFER, 4))
    
    x = conv_batchnorm_leaky_layer(x = input_image, neurons = 32, use_maxpooling = True)
    x = conv_batchnorm_leaky_layer(x = x, neurons = 64, use_maxpooling = True)
    x = conv_batchnorm_leaky_layer(x = x, neurons = 128)
    x = conv_batchnorm_leaky_layer(x = x, neurons = 64, kernel = (1,1), use_maxpooling = True)
    x = conv_batchnorm_leaky_layer(x = x, neurons = 128, use_maxpooling = True)
    x = conv_batchnorm_leaky_layer(x = x, neurons = 256)
    x = conv_batchnorm_leaky_layer(x = x, neurons = 128, use_maxpooling = True)
    x = conv_batchnorm_leaky_layer(x = x, neurons = 256)
    x = Conv2D(BOX * (4 + 1 + CLASS), (1,1), strides=(1,1), padding='same', name='conv_23')(x)
    output = Reshape((GRID_H, GRID_W, BOX, 4 + 1 + CLASS))(x)
    output = Lambda(lambda args: args[0])([output, true_boxes])
    model = Model([input_image, true_boxes], output)
    return model



def yolo_reorg_convnet(image_h = generator_config.get('IMAGE_H'),
                    image_w = generator_config.get('IMAGE_W'),
                    true_box_buffer = generator_config.get('TRUE_BOX_BUFFER'),
                    anchors = generator_config.get('ANCHORS'),
                    classes = generator_config.get('LABELS'),
                    grid_h = generator_config.get('GRID_H'),
                    grid_w = generator_config.get('GRID_W'),
                    box = generator_config.get('BOX'),
                    n_channels = 3,
                    pool_size = (2,2)):
    """https://github.com/guigzzz/Keras-Yolo-v2/blob/master/yolo_v2.py"""
    input_image = Input(shape=(image_h, image_w, 3))
    #true_boxes  = Input(shape=(1, 1, 1, true_box_buffer, 4))
    
    x = conv_batchnorm_leaky_layer(x = input_image, neurons = 32, use_maxpooling = True)
    x = conv_batchnorm_leaky_layer(x = x, neurons = 64, use_maxpooling = True)
    
    # Network in Network 1
    x = network_in_network(x, [(128, 3), (64, 1), (128, 3)])
    x = MaxPooling2D(pool_size = pool_size)(x)
    
    # Network in Network 2
    x = network_in_network(x, [(256, 3), (128, 1), (256, 3)])
    x = MaxPooling2D(pool_size = pool_size)(x)
    
    # Network in Network 3
    x = network_in_network(x, [(512, 3), (256, 1), (512, 3), (256, 1), (512, 3)])
    skip = x
    x = MaxPooling2D(pool_size = pool_size)(x)
    
    # Network in Network 4
    x = network_in_network(x, [(1024, 3), (512, 1), (1024, 3), (512, 1), (1024, 3)])
    
    # Detection Layers
    x = conv_batchnorm_leaky_layer(x = x, neurons = 1024)
    x = conv_batchnorm_leaky_layer(x = x, neurons = 1024)
    
    # Concatenate Skip Connection
    x_skip = conv_batchnorm_leaky_layer(x = skip, neurons = 64)
    x = Concatenate()[reorg(x_skip, 2), x]
    
    # 
    x = conv_batchnorm_leaky_layer(x = x, neurons = 1024)
    
    n_outputs = len(anchors) * (5 + len(classes))
    x = Conv2D(n_outputs, (1, 1), padding = 'same', activation = 'linear')(x)
    model_out = Reshape([grid_h, grid_w, box, 4 + 1 + len(classes)])(x)
    return Model(inputs = input_image,  outputs = model_out)



    
    
### Prediction Functions-
###############################################################################
    
class OutputRescaler(object):
    def __init__(self, ANCHORS = generator_config['ANCHORS']):
        self.ANCHORS = ANCHORS

    def _sigmoid(self, x):
        return 1. / (1. + np.exp(-x))
    
    def _softmax(self, x, axis=-1, t=-100.):
        x = x - np.max(x)
        if np.min(x) < t:
            x = x/np.min(x)*t
        e_x = np.exp(x)
        return e_x / e_x.sum(axis, keepdims=True)
    
    def get_shifting_matrix(self,netout):
        
        GRID_H, GRID_W, BOX = netout.shape[:3]
        no = netout[...,0]
        
        ANCHORSw = self.ANCHORS[::2]
        ANCHORSh = self.ANCHORS[1::2]
       
        mat_GRID_W = np.zeros_like(no)
        for igrid_w in range(GRID_W):
            mat_GRID_W[:,igrid_w,:] = igrid_w

        mat_GRID_H = np.zeros_like(no)
        for igrid_h in range(GRID_H):
            mat_GRID_H[igrid_h,:,:] = igrid_h

        mat_ANCHOR_W = np.zeros_like(no)
        for ianchor in range(BOX):    
            mat_ANCHOR_W[:,:,ianchor] = ANCHORSw[ianchor]

        mat_ANCHOR_H = np.zeros_like(no) 
        for ianchor in range(BOX):    
            mat_ANCHOR_H[:,:,ianchor] = ANCHORSh[ianchor]
        return(mat_GRID_W,mat_GRID_H,mat_ANCHOR_W,mat_ANCHOR_H)

    def fit(self, netout):    
        '''
        netout  : np.array of shape (N grid h, N grid w, N anchor, 4 + 1 + N class)
        
        a single image output of model.predict()
        '''
        GRID_H, GRID_W, BOX = netout.shape[:3]
        
        (mat_GRID_W,
         mat_GRID_H,
         mat_ANCHOR_W,
         mat_ANCHOR_H) = self.get_shifting_matrix(netout)


        # bounding box parameters
        netout[..., 0]   = (self._sigmoid(netout[..., 0]) + mat_GRID_W)/GRID_W # x      unit: range between 0 and 1
        netout[..., 1]   = (self._sigmoid(netout[..., 1]) + mat_GRID_H)/GRID_H # y      unit: range between 0 and 1
        netout[..., 2]   = (np.exp(netout[..., 2]) * mat_ANCHOR_W)/GRID_W      # width  unit: range between 0 and 1
        netout[..., 3]   = (np.exp(netout[..., 3]) * mat_ANCHOR_H)/GRID_H      # height unit: range between 0 and 1
        # rescale the confidence to range 0 and 1 
        netout[..., 4]   = self._sigmoid(netout[..., 4])
        expand_conf      = np.expand_dims(netout[...,4],-1) # (N grid h , N grid w, N anchor , 1)
        # rescale the class probability to range between 0 and 1
        # Pr(object class = k) = Pr(object exists) * Pr(object class = k |object exists)
        #                      = Conf * P^c
        netout[..., 5:]  = expand_conf * self._softmax(netout[..., 5:])
        # ignore the class probability if it is less than obj_threshold 
    
        return(netout)


def find_high_class_probability_bbox(netout_scale, obj_threshold):
    '''
    == Input == 
    netout : y_pred[i] np.array of shape (GRID_H, GRID_W, BOX, 4 + 1 + N class)
    
             x, w must be a unit of image width
             y, h must be a unit of image height
             c must be in between 0 and 1
             p^c must be in between 0 and 1
    == Output ==
    
    boxes  : list containing bounding box with Pr(object is in class C) > 0 for at least in one class C 
    
             
    '''
    GRID_H, GRID_W, BOX = netout_scale.shape[:3]
    
    boxes = []
    for row in range(GRID_H):
        for col in range(GRID_W):
            for b in range(BOX):
                # from 4th element onwards are confidence and class classes
                classes = netout_scale[row,col,b,5:]
                
                if np.sum(classes) > 0:
                    # first 4 elements are x, y, w, and h
                    x, y, w, h = netout_scale[row,col,b,:4]
                    confidence = netout_scale[row,col,b,4]
                    box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, confidence, classes)
                    if box.get_score() > obj_threshold:
                        boxes.append(box)
    return(boxes)


def nonmax_suppression(boxes, iou_threshold, obj_threshold):
    '''
    boxes : list containing "good" BoundBox of a frame
            [BoundBox(),BoundBox(),...]
    '''
    bestAnchorBoxFinder    = BestAnchorBoxFinder([])
    if len(boxes) > 0:
        CLASS = len(boxes[0].classes)
    else:
        CLASS = 0
    index_boxes = []   
    # suppress non-maximal boxes
    for c in range(CLASS):
        # extract class probabilities of the c^th class from multiple bbox
        class_probability_from_bbxs = [box.classes[c] for box in boxes]

        #sorted_indices[i] contains the i^th largest class probabilities
        sorted_indices = list(reversed(np.argsort( class_probability_from_bbxs)))

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            
            # if class probability is zero then ignore
            if boxes[index_i].classes[c] == 0:  
                continue
            else:
                index_boxes.append(index_i)
                for j in range(i+1, len(sorted_indices)):
                    index_j = sorted_indices[j]
                    
                    # check if the selected i^th bounding box has high IOU with any of the remaining bbox
                    # if so, the remaining bbox' class probabilities are set to 0.
                    bbox_iou = bestAnchorBoxFinder.bbox_iou(boxes[index_i], boxes[index_j])
                    if bbox_iou >= iou_threshold:
                        classes = boxes[index_j].classes
                        classes[c] = 0
                        boxes[index_j].set_class(classes)
                        
    newboxes = [ boxes[i] for i in index_boxes if boxes[i].get_score() > obj_threshold ]                
    
    return newboxes




def plot_image_bounding_box(img_arr, boxes, labels = generator_config['LABELS'],
                            box_color = 'red', text_color = 'red', 
                            fontsize = 11, linewidth = 1, y_offset = -10, obj_baseline=0.05):
    """
    Create a matplotlib image plot with one or more bounding boxes
    
    Args:
        img_arr (numpy.array): 3d numpy array of image
        boxes (list): list of BoundBox() objects
        labels (list): list of class labels (defaults to generator_config)
        box_color (str): color to use in bounding box edge (defaults to 'red')
        text_color (str): color to use in text label (defaults to 'red')
        fontsize (int): size to use for label font (defaults to 11)
        linewidth (int): size to use for box edge line width (defaults to 1)
        y_offset (int): how far to offset text label from upper-left corner of bounding box (defaults to -10)
    """
    def adjust_minmax(c,_max):
        if c < 0:
            c = 0   
        if c > _max:
            c = _max
        return c
    
    image = copy.deepcopy(img_arr)
    image_h, image_w, _ = image.shape
    score_rescaled  = np.array([box.get_score() for box in boxes])
    score_rescaled /= obj_baseline
    
    colors = sns.color_palette("husl", 8)
    fig,ax = plt.subplots(1)
    
    for sr, box,color in zip(score_rescaled, boxes, colors):
        xmin = adjust_minmax(int(box.xmin*image_w),image_w)
        ymin = adjust_minmax(int(box.ymin*image_h),image_h)
        xmax = adjust_minmax(int(box.xmax*image_w),image_w)
        ymax = adjust_minmax(int(box.ymax*image_h),image_h)
        text = "{:10} {:4.3f}".format(labels[box.label], box.get_score())
        box_width = xmax - xmin
        box_height = ymax - ymin
        
        # Create rectangle and label text
        rect = patches.Rectangle((xmin, ymin), box_width, box_height, linewidth = linewidth, edgecolor = box_color, facecolor = 'none')
        ax.text(xmin, ymin + y_offset, text, color = text_color, fontsize = fontsize)
        ax.add_patch(rect)
        print(text)
    plt.imshow(img_arr)
    plt.show()
    




### Load Data
###############################################################################
# Load Data
with open(f'{dict_write_folder}{dict_list_save_name}', 'rb') as fp:
    train_image = pickle.load(fp)   



### Create Generator
###############################################################################
keras.backend.clear_session()
train_batch_generator = SimpleBatchGenerator(train_image, generator_config, shuffle=True)



### Choose Model Architecture, Optimizer, & Compile with Custom Loss
###############################################################################
# Declare Model Variable
model = yolo_v2_convnet()


# Load Weights
weight_reader = WeightReader(weights_path)
weight_reader.reset()
nb_conv = 21

for i in range(1, nb_conv+1):
    conv_layer = model.get_layer('conv_' + str(i))
    print(f'Finished {i}')
    
    if i < nb_conv:
        norm_layer = model.get_layer('norm_' + str(i))
        
        size = np.prod(norm_layer.get_weights()[0].shape)

        beta  = weight_reader.read_bytes(size)
        gamma = weight_reader.read_bytes(size)
        mean  = weight_reader.read_bytes(size)
        var   = weight_reader.read_bytes(size)

        weights = norm_layer.set_weights([gamma, beta, mean, var])       
        
    if len(conv_layer.get_weights()) > 1:
        bias   = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
        kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
        kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
        kernel = kernel.transpose([2,3,1,0])
        conv_layer.set_weights([kernel, bias])
    else:
        kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
        kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
        kernel = kernel.transpose([2,3,1,0])
        conv_layer.set_weights([kernel])












optimizer = Adam(lr=0.00001)



loss_yolo = lf.YoloLoss(generator_config['ANCHORS'], (generator_config['GRID_W'], generator_config['GRID_H']),
                        generator_config['BATCH_SIZE'],
                        lambda_coord=1.0,
                        lambda_noobj=1.0,
                        lambda_obj=5.0,
                        lambda_class=1.0)



#yolo_model.compile(loss = YoloLoss(), optimizer = optimizer)

model.compile(loss = loss_yolo, optimizer = optimizer)

# If we run this (even though it's wrong), we don't run out of memory
# yolo_model.compile(loss = 'categorical_crossentropy', optimizer = optimizer)


model.summary()















### Early Stopping Callbacks
###############################################################################
early_stop = keras.callbacks.EarlyStopping(monitor='loss',  min_delta = 0.001, patience = 1,  mode='min',  verbose=1)

checkpoint = keras.callbacks.ModelCheckpoint(f'{model_save_path}{model_save_name}', 
                             monitor='loss', 
                             verbose=1, 
                             save_best_only=True, 
                             mode='min')

lr_schedule = m.CyclicalRateSchedule(min_lr = 0.0000125,
                                     max_lr = 0.00015,
                                     n_epochs = 10,
                                     warmup_epochs = 5,
                                     cooldown_epochs = 1,
                                     cycle_length = 10,
                                     logarithmic = True,
                                     decrease_factor = 0.9)


### Fit Model
###############################################################################
# Must Be Run with Eager Execution
tf.config.run_functions_eagerly(True)

# Fit
model.fit(train_batch_generator, 
               steps_per_epoch  = len(train_batch_generator),
               epochs = 50, 
               verbose = 1,
               callbacks = [early_stop, checkpoint])




### Make Prediction
###############################################################################




        
"""

random_image_name = random.choice(os.listdir('C:/local_images/'))
img_reader = ImageReader()
test_image = img_reader.fit(f'C:/local_images/{random_image_name}')
test_image = np.expand_dims(test_image, 0)
dummy_array = np.zeros((1, 1, 1, 1, generator_config['TRUE_BOX_BUFFER'], 4))
test_pred = yolo_model.predict([test_image, dummy_array])



netout = test_pred[0]
output_rescaler = OutputRescaler()
netout_scale = output_rescaler.fit(netout)


obj_threshold = 0.04
boxes = find_high_class_probability_bbox(netout_scale,obj_threshold)
    
iou_threshold = 0.01
final_boxes = nonmax_suppression(boxes, iou_threshold = iou_threshold, obj_threshold = obj_threshold)
print("{} final number of boxes".format(len(final_boxes)))



plot_image_bounding_box(test_image[0], final_boxes, generator_config['LABELS'])      
        
        
        




Epoch 1/5
loss_xywh: 1.5628384351730347 ... type: <class 'tensorflow.python.framework.ops.EagerTensor'>

loss_conf: Tensor("RealDiv_3:0", shape=(), dtype=float32) ... type: <class 'tensorflow.python.framework.ops.Tensor'>

loss_class: 1.5242124795913696 ... type: <class 'tensorflow.python.framework.ops.EagerTensor'>

loss: Tensor("AddV2_5:0", shape=(), dtype=float32)



"""














