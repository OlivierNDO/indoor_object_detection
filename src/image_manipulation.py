### Configuration
###############################################################################

# Import Modules
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
import requests
import tempfile
from tensorflow.keras.preprocessing.image import load_img
import time
from sklearn.model_selection import train_test_split
from skimage.transform import resize
import tqdm

# Import Project Modules
import src.config_data_processing as cdp
import src.misc_functions as mf

### Define Functions
###############################################################################


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

 
def load_resize_images_from_files(full_file_paths, resize_height, resize_width):
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


def load_resize_images_from_urls(url_list, resize_height, resize_width):
    """
    Load images from list of URLs and resize according to function arguments
    
    Args:
        url_list: list of image URLs
        resize_height: height of resized output images
        resize_width: width of resized output images
    Depdendencies:
        numpy
        skimage.transform.resize
        tensorflow.keras.preprocessing.image.load_img
    Returns:
        4d numpy array of resized images
    """
    read_images = []
    for i, x in tqdm.tqdm(enumerate(url_list)):
        try:
            img = read_url_image(x)
            resized_img = resize(img[:,:,:3], (resize_height, resize_width))
            read_images.append(resized_img)
        except:
            read_images.append(np.empty((resize_width, resize_height, 3)))
    return read_images


def extract_image_subset(img_arr, xmin, xmax, ymin, ymax, decimal = True):
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
    else:
        output = img_arr[ymin:ymax, xmin:xmax]
    return output


def plot_image_bounding_box(img_arr, xmin, xmax, ymin, ymax, label,
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
    for i, x in enumerate(xmin):
        
        xmin_p = int(x * w)
        xmax_p = int(xmax[i] * w)
        ymin_p = int(ymin[i] * h)
        ymax_p = int(ymax[i] * h)
        box_width = xmax_p - xmin_p
        box_height = ymax_p - ymin_p
    
        # Create rectangle and label text
        rect = patches.Rectangle((xmin_p, ymin_p), box_width, box_height, linewidth = linewidth, edgecolor = box_color, facecolor = 'none')
        ax.text(xmin_p, ymin_p + y_offset, label[i], color = text_color, fontsize = fontsize)
        ax.add_patch(rect)
    plt.imshow(img_arr)
    plt.show()




class OpenCVImageClassRetriever:
    """
    Retrieve, process, and save images from Google Open Images V6 URLs
    
    Args:
        class_name (str): class name corresponding to subset of images
        bucket_name (str): Google Storage Bucket to read csv files from
        img_csv_path (str): csv file path to train images information (contains URLs, image IDs, etc.)
        annotation_csv_path (str): csv file path to annotation file (object label information)
        bbox_csv_path (str): csv file path to bounding box coordinate file (bounding box dimensions)
        class_desc_csv_path (str): csv file path class description mapping
        image_id_col (str): column name convention in Google Open Images used for the image identifier
        image_url_col (str): column name convention in Google Open Images used for URL reading
    """
    
    def __init__(self, 
                 class_name,
                 local_gcs_json_path = f'{cdp.config_gcs_auth_json_path}',
                 bucket_name = f'{cdp.config_source_bucket_name}',
                 processed_bucket_subfolder = f'{cdp.config_processed_bucket_subfolder}',
                 processed_array_save_name = 'train_images.npy',
                 processed_bbox_save_name = 'train_bbox.csv',
                 processed_class_save_name = 'train_class_df.csv',
                 img_csv_path = f'{cdp.config_train_image_csv}',
                 annotation_csv_path = f'{cdp.config_train_annotations_csv}',
                 bbox_csv_path = f'{cdp.config_train_bbox_csv}',
                 class_desc_csv_path = f'{cdp.config_class_desc_csv}',
                 image_id_col = 'ImageID',
                 image_url_col = 'OriginalURL',
                 resize_height = cdp.config_resize_height,
                 resize_width = cdp.config_resize_width,
                 max_images = 5000,
                 ):
        # Initialize Arguments
        self.class_name = class_name
        self.local_gcs_json_path = local_gcs_json_path
        self.bucket_name = bucket_name
        self.processed_bucket_subfolder = processed_bucket_subfolder
        self.processed_array_save_name = processed_array_save_name
        self.processed_bbox_save_name = processed_bbox_save_name
        self.processed_class_save_name = processed_class_save_name
        self.img_csv_path = img_csv_path
        self.annotation_csv_path = annotation_csv_path
        self.bbox_csv_path = bbox_csv_path
        self.class_desc_csv_path = class_desc_csv_path
        self.image_id_col = image_id_col
        self.image_url_col = image_url_col
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.max_images = max_images
        
        # Reference Google Cloud Authentication Document
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.local_gcs_json_path
        

    def get_image_class_info(self):
        # Read Csv Files
        img_df = mf.read_gcs_csv_to_pandas(bucket_name = self.bucket_name, file_name = self.img_csv_path)
        annot_df = mf.read_gcs_csv_to_pandas(bucket_name = self.bucket_name, file_name = self.annotation_csv_path)
        class_desc_df = mf.read_gcs_csv_to_pandas(bucket_name = self.bucket_name, file_name = self.class_desc_csv_path, header = None)
        class_bbox_df = mf.read_gcs_csv_to_pandas(bucket_name = self.bucket_name, file_name = self.bbox_csv_path)
        
        # Create Dictionary of Image Class Labels
        class_label_dict = dict(zip(class_desc_df[1], class_desc_df[0]))
        
        # Retrieve and return URLs, Bounding Box DataFrame, and Image IDs
        class_label = class_label_dict.get(self.class_name)
        class_image_ids = list(annot_df[(annot_df.LabelName == class_label) & (annot_df.Confidence == 1)][self.image_id_col])
        if self.max_images is not None:
            class_image_ids = class_image_ids[:self.max_images]
        class_image_df = img_df[img_df.ImageID.isin(class_image_ids)]
        class_bbox_df = class_bbox_df[(class_bbox_df.ImageID.isin(class_image_ids)) & \
                                      (class_bbox_df.LabelName == class_label)]
        class_image_ids = list(class_bbox_df[self.image_id_col])
        class_image_urls = list(class_image_df[self.image_url_col])
        return class_image_urls, class_bbox_df, class_image_ids, class_image_df
    
    def resize_and_save_images(self):
        # Generate Class Information
        mf.print_timestamp_message(f'Getting urls, bounding boxes, and image IDs for {self.class_name} images')
        urls, bbox_df, image_ids, class_image_df = self.get_image_class_info()
        
        # Read and Resize Images
        n_url = len(urls)
        mf.print_timestamp_message(f'Reading images from {n_url} URLs and resizing to {self.resize_height} X {self.resize_width}')
        image_arrays = load_resize_images_from_urls(url_list = urls, resize_height = self.resize_height, resize_width = self.resize_width)
        image_arrays_concat = np.array(image_arrays)
        
        # Write Images to Google Cloud Storage Bucket
        image_save_name = f'{self.processed_bucket_subfolder}{self.class_name}/{self.processed_array_save_name}'
        mf.print_timestamp_message(f'Writing images to GCS bucket/folder {self.bucket_name}/{image_save_name}')
        mf.save_np_array_to_gsc(np_array = image_arrays_concat, bucket_name = self.bucket_name, file_name = image_save_name)
        
        # Write Bounding Box Csv to Google Cloud Storage Bucket
        bbox_save_name = f'{self.processed_bucket_subfolder}{self.class_name}/{self.processed_bbox_save_name}'
        mf.print_timestamp_message(f'Writing bounding box csv file to GCS bucket/folder {self.bucket_name}/{bbox_save_name}')
        mf.write_csv_to_gcs(dframe = bbox_df, bucket_name = self.bucket_name, file_name = bbox_save_name)
        
        # Write Class Info Csv to Google Cloud Storage Bucket
        class_save_name = f'{self.processed_bucket_subfolder}{self.class_name}/{self.processed_class_save_name}'
        mf.print_timestamp_message(f'Writing class image info csv file to GCS bucket/folder {self.bucket_name}/{class_save_name}')
        mf.write_csv_to_gcs(dframe = class_image_df, bucket_name = self.bucket_name, file_name = class_save_name)
        
    def get_class_image_array(self):
        # Assert Data Exists in GCS Bucket
        folder_exists = mf.gcs_subfolder_exists(bucket_name = self.bucket_name, subfolder_name = self.class_name)
        assert_msg = f"Data for image class '{self.class_name} doesn't exist. Create it with self.resize_and_save_images() method'"
        assert folder_exists, assert_msg
        
        # Read and Return Images
        image_save_name = f'{self.processed_bucket_subfolder}{self.class_name}/{self.processed_array_save_name}'
        img_array = mf.read_gcs_numpy_array(bucket_name = self.bucket_name, file_name = image_save_name)
        return img_array
    
    def get_bounding_box_df(self):
        # Assert Data Exists in GCS Bucket
        folder_exists = mf.gcs_subfolder_exists(bucket_name = self.bucket_name, subfolder_name = self.class_name)
        assert_msg = f"Data for image class '{self.class_name} doesn't exist. Create it with self.resize_and_save_images() method'"
        assert folder_exists, assert_msg
        
        # Read and Return Images
        csv_save_name = f'{self.processed_bucket_subfolder}{self.class_name}/{self.processed_bbox_save_name}'
        bbox_df = mf.read_gcs_csv_to_pandas(bucket_name = self.bucket_name, file_name = csv_save_name)
        return bbox_df
    
    def get_class_desc_df(self):
        # Assert Data Exists in GCS Bucket
        folder_exists = mf.gcs_subfolder_exists(bucket_name = self.bucket_name, subfolder_name = self.class_name)
        assert_msg = f"Data for image class '{self.class_name} doesn't exist. Create it with self.resize_and_save_images() method'"
        assert folder_exists, assert_msg
        
        # Read and Return Images
        csv_save_name = f'{self.processed_bucket_subfolder}{self.class_name}/{self.processed_class_save_name}'
        class_desc_df = mf.read_gcs_csv_to_pandas(bucket_name = self.bucket_name, file_name = csv_save_name)
        return class_desc_df
    
    def get_training_data(self):
        # Retrieve Class Image Array, Bounding Box DataFrame, and Description DataFrame
        img_arr = self.get_class_image_array()
        bbox_df = self.get_bounding_box_df()
        desc_df = self.get_class_desc_df()
        
        # Multiply Bounding Box Floats to Get Pixel Positions
        bbox_df_copy = bbox_df.copy()
        bbox_df_copy['XMin'] = [int(x * img_arr.shape[-2]) for x in bbox_df['XMin']]
        bbox_df_copy['XMax'] = [int(x * img_arr.shape[-2]) for x in bbox_df['XMax']]
        bbox_df_copy['YMin'] = [int(x * img_arr.shape[-2]) for x in bbox_df['YMin']]
        bbox_df_copy['YMax'] = [int(x * img_arr.shape[-2]) for x in bbox_df['YMax']]
        
        # Generate Training Datasets (X and Y)
        unique_image_ids = list(set(desc_df['ImageID']))
        image_ids = list(bbox_df_copy['ImageID'])
        y_array_list = mf.unnest_list_of_lists([bbox_df_copy[bbox_df_copy.ImageID == x][['XMin', 'XMax', 'YMin', 'YMax']].values.tolist() for x in unique_image_ids])
        img_array_positions = mf.unnest_list_of_lists([[i for i,x in enumerate(unique_image_ids) if x == z] for z in image_ids])
        train_x = np.array([img_arr[i] for i in img_array_positions])
        train_y = np.array(y_array_list)
        return train_x, train_y


def make_class_weight_dict(train_y_labels, return_dict = False):
    """
    Return dictionary of inverse class weights for imbalanced response
    
    Args:
        train_y_labels: training set response variable (list or numpy array)
        return_dict: if True, return dictionary of classes & weights..else return list of classes and list of weights
    """
    
    if str(type(train_y_labels)) == "<class 'numpy.ndarray'>":
        labs = list(range(train_y_labels.shape[1]))
        freq = list(np.sum(train_y_labels, axis = 0))
        train_class_counts = dict(zip(labs, freq))
    else:
        train_class_counts = dict((x,train_y_labels.count(x)) for x in set(train_y_labels))
    max_class = max(train_class_counts.values())
    class_weights = [max_class / x for x in train_class_counts.values()]
    class_weight_dict = dict(zip([i for i in train_class_counts.keys()], class_weights))
    if return_dict:
        return class_weight_dict
    else:
        return list(class_weight_dict.keys()), list(class_weight_dict.values())


def remove_blank_images(x_arr, y_arr, bbox_arr):
    """
    For ordered pair of x and y arrays, remove arrays with blank images in X
    
    Args:
        x_arr (numpy.array): 4d numpy array (images)
        y_arr (numpy.array): array or nested list with dependent variable
        bbox_arr (numpy.array): array of bounding box coordinates
    Returns:
        x (numpy.array), y (numpy.array), bbox (numpy array)
    """
    nan = [i for i, x in enumerate(x_arr) if np.isnan(np.sum(x))]
    inf = [i for i, x in enumerate(x_arr) if math.isinf(np.sum(x))]
    near_zero = [i for i, x in enumerate(x_arr) if (np.sum(x == 0) / np.sum(x != 0)) > 5]
    remove = list(set(nan + inf + near_zero))
    keep = [i for i in range(x_arr.shape[0]) if i not in remove]
    return x_arr[keep], y_arr[keep], bbox_arr[keep]


class OpenCVMultiClassProcessor:
    """
    Wrapper around OpenCVImageClassRetriever() class to retrieve and process
    images and bounding boxes from multiple classes
    """
    
    def __init__(self, 
                 class_list,
                 max_images = None,
                 random_state = 9242020,
                 resize_height = cdp.config_resize_height,
                 resize_width = cdp.config_resize_width,
                 test_size = 0.2):
        # Initialize Arguments
        self.class_list = class_list
        self.max_images = max_images
        self.random_state = random_state
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.test_size = test_size
        
    
    def get_processed_data(self):
        # Retrieve Images, Classification Array, and Bounding Boxes for Multiple Classes
        x_img_list = []
        y_bbox_list = []
        y_classif_list = []
        
        # Loop Over Image CLasses to Retrieve Data
        for i, x in enumerate(self.class_list):
            mf.print_timestamp_message(f"Pulling data for class '{x}' ({i+1} of {len(self.class_list)}) from Google Cloud Storage")
            image_retriever = OpenCVImageClassRetriever(class_name = x)
            x_img, y_bbox = image_retriever.get_training_data()
            x_img_list.append(x_img)
            y_bbox_list.append(y_bbox)
            y_classif_list.append([x] * x_img.shape[0])
        
        # Concatenate / Unnest Outer Lists
        x_img_list = np.vstack(x_img_list)
        y_bbox_list = np.vstack(y_bbox_list)
        y_classif_list = mf.unnest_list_of_lists(y_classif_list)
        return x_img_list, y_bbox_list, y_classif_list
    
    def get_train_test_valid_data(self):
        # Retrieve Images, Classification Array, and Bounding Boxes for Multiple Classes
        x_img_list, y_bbox_list, y_classif_list = self.get_processed_data()
        
        # Split into Train and Test
        train_x, test_x, \
        train_y_bbox, test_y_bbox, \
        train_y_classif, test_y_classif = train_test_split(x_img_list, y_bbox_list, y_classif_list,
                                                           test_size = self.test_size * 2,
                                                           random_state = self.random_state)
        # Split Test in Half -> Test & Validation
        valid_x, test_x, \
        valid_y_bbox, test_y_bbox, \
        valid_y_classif, test_y_classif = train_test_split(test_x, test_y_bbox, test_y_classif,
                                                           test_size = 0.5,
                                                           random_state = self.random_state)
        
        # Create Class Weight Dictionary & Convert Strings in Y to Numbers
        class_list, class_weights = make_class_weight_dict(list(train_y_classif), return_dict = False)
        class_weight_dict = dict(zip(list(range(len(class_weights))), class_weights))
        class_list_int_dict = dict(zip(class_list, list(range(len(class_list)))))
        train_y_classif = np.vstack([class_list_int_dict.get(s) for s in train_y_classif])
        test_y_classif = np.vstack([class_list_int_dict.get(s) for s in test_y_classif])
        valid_y_classif = np.vstack([class_list_int_dict.get(s) for s in valid_y_classif])
        
        # Remove Blank Images
        train_x, train_y, train_bbox = remove_blank_images(train_x[:self.max_images], train_y_classif[:self.max_images],  train_y_bbox[:self.max_images])
        test_x, test_y, test_bbox = remove_blank_images(test_x[:self.max_images], test_y_classif[:self.max_images],  test_y_bbox[:self.max_images])
        valid_x, valid_y, valid_bbox = remove_blank_images(valid_x[:self.max_images], valid_y_classif[:self.max_images],  valid_y_bbox[:self.max_images])
        
        # One-Hot Encode Y-Values
        train_y = np.array([[1 if t == i else 0 for i, x in enumerate(np.unique(train_y))] for t in train_y])
        test_y = np.array([[1 if t == i else 0 for i, x in enumerate(np.unique(test_y))] for t in test_y])
        valid_y = np.array([[1 if t == i else 0 for i, x in enumerate(np.unique(valid_y))] for t in valid_y])
        
        
        # Generate Object-Only Crops of X Arrays: Train
        train_obj_x = []
        for i in range(train_bbox.shape[0]):
            t = extract_image_subset(train_x[i], train_bbox[i][0], train_bbox[i][1], train_bbox[i][2], train_bbox[i][3], decimal = False)
            train_obj_x.append(resize(t, (self.resize_width, self.resize_height)))
        train_obj_x = np.array(train_obj_x)
        
        # Generate Object-Only Crops of X Arrays: Test
        test_obj_x = []
        for i in range(test_bbox.shape[0]):
            t = extract_image_subset(test_x[i], test_bbox[i][0], test_bbox[i][1], test_bbox[i][2], test_bbox[i][3], decimal = False)
            test_obj_x.append(resize(t, (self.resize_width, self.resize_height)))
        test_obj_x = np.array(test_obj_x)
        
        
        # Generate Object-Only Crops of X Arrays: Validation
        valid_obj_x = []
        for i in range(valid_bbox.shape[0]):
            t = extract_image_subset(valid_x[i], valid_bbox[i][0], valid_bbox[i][1], valid_bbox[i][2], valid_bbox[i][3], decimal = False)
            valid_obj_x.append(resize(t, (self.resize_width, self.resize_height)))
        valid_obj_x = np.array(valid_obj_x)
        
        

        # Create and Return Dictionary
        dict_keys = ['TRAIN X', 'TEST X', 'VALIDATION X',
                     'TRAIN OBJECT X', 'TEST OBJECT X', 'VALIDATION OBJECT X',
                     'TRAIN BBOX', 'TEST BBOX', 'VALIDATION BBOX',
                     'TRAIN Y', 'TEST Y', 'VALIDATION Y',
                     'CLASS WEIGHT DICT']
        dict_values = [train_x, test_x, valid_x,
                       train_obj_x, test_obj_x, valid_obj_x,
                       train_bbox, test_bbox, valid_bbox,
                       train_y, test_y, valid_y, class_weight_dict]
        return dict(zip(dict_keys, dict_values))



def is_blank_img(img_arr, nonzero_threshold = 0.2):
    nonzero_percent = np.count_nonzero(img_arr) / np.product(img_arr.shape)
    return any([np.isnan(np.sum(img_arr)), math.isinf(np.sum(img_arr)), nonzero_percent < nonzero_threshold])




class OpenCVCroppedImageRetriever:
    """
    Retrieve, process, and crop via bounding box before saving images from Google Open Images V6 URLs
    
    Args:
        class_name (str): class name corresponding to subset of images
    """
    
    def __init__(self, 
                 class_name,
                 local_gcs_json_path = f'{cdp.config_gcs_auth_json_path}',
                 image_id_col = 'ImageID',
                 bucket_name = f'{cdp.config_source_bucket_name}',
                 processed_bucket_subfolder = f'{cdp.config_processed_bucket_subfolder}',
                 processed_array_save_name = 'train_images_cropped.npy',
                 resize_height = cdp.config_resize_height,
                 resize_width = cdp.config_resize_width,
                 max_images = 5000
                 ):
        # Initialize Arguments
        self.class_name = class_name
        self.local_gcs_json_path = local_gcs_json_path
        self.image_id_col = image_id_col
        self.local_gcs_json_path = local_gcs_json_path
        self.bucket_name = bucket_name
        self.processed_bucket_subfolder = processed_bucket_subfolder
        self.processed_array_save_name = processed_array_save_name
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.max_images = max_images
        
        # Reference Google Cloud Authentication Document
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.local_gcs_json_path
        
    
    def get_cropped_obj_images(self):
        # Retrieve Class Metadata
        image_retriever = OpenCVImageClassRetriever(class_name = self.class_name)
        bbox_df = image_retriever.get_bounding_box_df()
        desc_df = image_retriever.get_class_desc_df()
        
        # Image IDs
        unique_img_ids = list(np.unique(bbox_df[self.image_id_col].values.tolist()))
        if self.max_images is not None:
            unique_img_ids = unique_img_ids[:self.max_images]
        
        # Read and Crop Images with Bounding Boxes
        cropped_img_list = []
        for img_id in tqdm.tqdm(unique_img_ids):
            try:
                # Subset Info Dataframes for Image ID
                bbox_df_i = bbox_df[bbox_df.ImageID == img_id]
                desc_df_i = desc_df[desc_df.ImageID == img_id]
                
                # Read Image
                img_i = read_url_image(desc_df_i['OriginalURL'].values[0])
            
                # Extract Cropped Objects
                bbox_coords = bbox_df_i[['XMin', 'XMax', 'YMin', 'YMax']].values.tolist()
                for bbc in bbox_coords:
                    xmin, xmax, ymin, ymax = bbc
                    img_subset = resize(extract_image_subset(img_i, xmin, xmax, ymin, ymax), (self.resize_width, self.resize_height))
                    correct_shape = (self.resize_width, self.resize_height, 3)
                    if (not is_blank_img(img_subset) and img_subset.shape == correct_shape):
                        cropped_img_list.append(img_subset)
            except:
                pass
        return np.array(cropped_img_list)
    
    
    def cropped_obj_images_to_gcs(self):
        # Read, Crop, and Resize Images
        mf.print_timestamp_message(f'Reading, cropping, and resizing {self.class_name} images')
        image_arrays_concat = self.get_cropped_obj_images()
        n_images = image_arrays_concat.shape[0]
        
        # Write Images to Google Cloud Storage Bucket
        image_save_name = f'{self.processed_bucket_subfolder}{self.class_name}/{self.processed_array_save_name}'
        mf.print_timestamp_message(f'Writing {n_images} cropped images to GCS bucket/folder {self.bucket_name}/{image_save_name}')
        mf.save_np_array_to_gsc(np_array = image_arrays_concat, bucket_name = self.bucket_name, file_name = image_save_name)
        
        
    def read_cropped_image_array(self):
        # Assert Data Exists in GCS Bucket
        folder_exists = mf.gcs_subfolder_exists(bucket_name = self.bucket_name, subfolder_name = self.class_name)
        assert_msg = f"Data for image class '{self.class_name} doesn't exist. Create it with self.cropped_obj_images_to_gcs() method'"
        assert folder_exists, assert_msg
        
        # Read and Return Images
        image_save_name = f'{self.processed_bucket_subfolder}{self.class_name}/{self.processed_array_save_name}'
        img_array = mf.read_gcs_numpy_array(bucket_name = self.bucket_name, file_name = image_save_name)
        return img_array







