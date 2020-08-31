### Overview
###############################################################################
# data source: 
#       > https://storage.googleapis.com/openimages/web/download.html
# helpful resources:
#       > https://github.com/RockyXu66/Faster_RCNN_for_Open_Images_Dataset_Keras/blob/master/Object_Detection_DataPreprocessing.ipynb
#       > https://machinelearningmastery.com/how-to-perform-object-detection-with-yolov3-in-keras/


### Configuration
###############################################################################
# Import Modules
import collections
import datetime
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
import numpy as np
import os
import pandas as pd
from PIL import Image
import requests
from tensorflow.keras.preprocessing.image import load_img
import time
from skimage.transform import resize

# File Paths
config_data_dir = 'D:/indoor_object_detection/data/source_data/'
config_proc_data_dir = 'D:/indoor_object_detection/data/processed_data/'
config_train_image_csv = 'train-images-boxable-with-rotation.csv'
config_train_annotations_csv = 'train-annotations-human-imagelabels-boxable.csv'
config_train_bbox_csv = 'oidv6-train-annotations-bbox.csv'
config_class_desc_csv = 'class-descriptions-boxable.csv'


### Define Functions
###############################################################################


def get_unique_count_dict(lst):
    """
    Generate and return dictionary with counts of unique items in a list
    Args:
        lst (list): list for which to generate unique element counts
    """
    key_values = collections.Counter(lst).keys()
    count_values = collections.Counter(lst).values()
    return dict(zip(key_values, count_values))


def img_add_flip(arr, flip_horiz = True, flip_vert = False):
    """
    Flip numpy array horizontally and/or vertically
    Args:
        arr: three dimensional numpy array
        flip_horiz: flip image horizontally
        flip_vert: flip image vertically
    """
    assert len(arr.shape) == 3, "'arr' input array must be three dimensional"
    arr_copy = arr.copy()
    if flip_horiz:
        arr_copy = np.fliplr(arr_copy)
    if flip_vert:
        arr_copy = np.flipud(arr_copy)
    return arr_copy


def load_resize_images(full_file_paths, resize_height, resize_width):
    """
    Load images and resize according to function arguments
    Args:
        full_file_paths: list of saved image files
        resize_height: height of resized output images
        resize_width: width of resized output images
    Depdendencies:
        numpy
        skimage.transform.resize
        tensorflow.keras.preprocessing.image.load_img
    Returns:
        numpy array of resized images
    """
    read_images = [load_img(c) for c in full_file_paths]
    resized_images = [resize(np.array(ri), (resize_height, resize_width)) for ri in read_images]
    return np.array(resized_images)


def read_url_image(url):
    """
    Read image from URL with .jpg or .png extension
    Args:
        url (str): url character string
    Returns:
        numpy array
    """
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return np.array(img)


def print_timestamp_message(message, timestamp_format = '%Y-%m-%d %H:%M:%S'):
    """
    Print formatted timestamp followed by custom message
    Args:
        message (str): string to concatenate with timestamp
        timestamp_format (str): format for datetime string. defaults to '%Y-%m-%d %H:%M:%S'
    """
    ts_string = datetime.datetime.fromtimestamp(time.time()).strftime(timestamp_format)
    print(f'{ts_string}: {message}')


def create_folder_if_not_existing(folder_path):
    """
    Use os to create a folder if it does not already exist on the machien
    Args:
        folder_path (str): folder path to conditionally create
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print_timestamp_message(f'Creating folder path {folder_path}')
    else:
        print_timestamp_message(f'Folder path {folder_path} already exists')
        
        



### Execute Functions
###############################################################################
#os.listdir(config_data_dir)
train_img_df = pd.read_csv(f'{config_data_dir}{config_train_image_csv}')
train_annot_df = pd.read_csv(f'{config_data_dir}{config_train_annotations_csv}')
class_desc_df = pd.read_csv(f'{config_data_dir}{config_class_desc_csv}', header = None)
class_label_dict = dict(zip(class_desc_df[1], class_desc_df[0]))
    
    
example_class = 'Loveseat'
example_label = class_label_dict.get(example_class)
example_class_image_ids = list(train_annot_df[(train_annot_df.LabelName == example_label) & (train_annot_df.Confidence == 1)]['ImageID'])
example_class_image_df = train_img_df[train_img_df.ImageID.isin(example_class_image_ids)]
example_class_image_urls = list(example_class_image_df['OriginalURL'])
example_class_bbox_df = pd.read_csv(f'{config_data_dir}{config_train_bbox_csv}')
example_class_bbox_df = example_class_bbox_df[example_class_bbox_df.ImageID.isin(example_class_image_ids)]



class OpenCVImageClassRetriever:
    def __init__(self, 
                 class_name,
                 img_csv_path = f'{config_data_dir}{config_train_image_csv}',
                 annotation_csv_path = f'{config_data_dir}{config_train_annotations_csv}',
                 bbox_csv_path = f'{config_data_dir}{config_train_bbox_csv}',
                 class_desc_csv_path = f'{config_data_dir}{config_class_desc_csv}',
                 output_path = config_proc_data_dir,
                 image_id_col = 'ImageID',
                 image_url_col = 'OriginalURL'
                 ):
        self.class_name = class_name
        self.img_csv_path = img_csv_path
        self.annotation_csv_path = annotation_csv_path
        self.bbox_csv_path = bbox_csv_path
        self.class_desc_csv_path = class_desc_csv_path
        self.output_path = output_path
        self.image_id_col = image_id_col
        self.image_url_col = image_url_col
    """
    Retrieve, process, and save images from Google Open Images V6 URLs
    Args:
        class_name (str): class name corresponding to subset of images
        img_csv_path (str): csv file path to train images information (contains URLs, image IDs, etc.)
        annotation_csv_path (str): csv file path to annotation file (object label information)
        bbox_csv_path (str): csv file path to bounding box coordinate file (bounding box dimensions)
        class_desc_csv_path (str): csv file path class description mapping
        output_path (str): folder path where output is to be written
        image_id_col (str): column name convention in Google Open Images used for the image identifier
        image_url_col (str): column name convention in Google Open Images used for URL reading
    """
    
    def get_image_class_info(self):
        # Read Csv Files
        img_df = pd.read_csv(self.img_csv_path)
        annot_df = pd.read_csv(self.annotation_csv_path)
        class_desc_df = pd.read_csv(self.class_desc_csv_path, header = None)
        class_bbox_df = pd.read_csv(self.bbox_csv_path)
        
        # Create Dictionary of Image Class Labels
        class_label_dict = dict(zip(class_desc_df[1], class_desc_df[0]))
        
        # Retrieve and return URLs, Bounding Box DataFrame, and Image IDs
        class_label = class_label_dict.get(self.class_name)
        class_image_ids = list(annot_df[(annot_df.LabelName == class_label) & (annot_df.Confidence == 1)][self.image_id_col])
        class_image_df = img_df[img_df.ImageID.isin(class_image_ids)]
        class_bbox_df = class_bbox_df[(class_bbox_df.ImageID.isin(class_image_ids)) & \
                                      (class_bbox_df.LabelName == class_label)]
        class_image_ids = list(class_bbox_df[self.image_id_col])
        class_image_urls = list(class_image_df[self.image_url_col])
        return class_image_urls, class_bbox_df, class_image_ids
    
    
    def resize_and_save_images(self):
        print('this function does not do anything yet')
    
    

def plot_image_bounding_box(img_array, xmin, xmax, ymin, ymax):
    """
    Create a matplotlib image plot with one or more bounding boxes
    Args:
        img_array (numpy.array): numpy array of image
        xmin (list): list of x-minimum coordinates (expressed as percentages)
        xmax (list): list of x-maximum coordinates (expressed as percentages)
        ymin (list): list of y-minimum coordinates (expressed as percentages)
        ymax (list): list of y-maximum coordinates (expressed as percentages)
    """
    # Extract image dimensions and create plot object
    h, w, c = img.shape
    fig,ax = plt.subplots(1)
    ax.imshow(img)
    
    for i, xM in enumerate(xmin):
        # Extract coordinates and dimensions
        xmin_p = int(xM * w)
        xmax_p = int(xmax[i] * w)
        ymin_p = int(ymin[i] * h)
        ymax_p = int(ymax[i] * h)
        coords = [xmin_p, xmax_p, ymin_p, ymax_p]
        print(coords)
        upper_left_point = (xmin_p, ymax_p)
        box_width = xmax_p - xmin_p
        box_height = ymax_p - ymin_p
        print(upper_left_point)
        
        # Create rectangle
        rect = patches.Rectangle((xmin_p, ymax_p), box_width, box_height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()



image_retriever = OpenCVImageClassRetriever(class_name = 'Loveseat')
temp_urls, temp_bbox_df, image_ids = image_retriever.get_image_class_info()


i = 32
img_id = image_ids[i]
img_bbox = temp_bbox_df[temp_bbox_df.ImageID == img_id]
img = read_url_image(temp_urls[i])



plot_image_bounding_box(img_array = img,
                        xmin = list(img_bbox['XMin']),
                        xmax = list(img_bbox['XMax']),
                        ymin = list(img_bbox['YMin']),
                        ymax = list(img_bbox['YMax']))






