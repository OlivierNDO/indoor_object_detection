### Configuration
###############################################################################
# Import Python Modules
import collections
import datetime
from google.cloud import storage
from io import BytesIO, StringIO
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
import tqdm

# Import Project Modules
import src.config_data_processing as cdp
import src.image_manipulation as imm
import src.misc_functions as mf



### Discuss Internal Modules (src folder)
###############################################################################

# config_data_processing



# image manipulation



# misc_functions





### Reading Source Files from Google Cloud Storage Bucket
###############################################################################

# Class Descriptions
class_description_df = mf.read_gcs_csv_to_pandas(bucket_name = cdp.config_source_bucket_name,
                                                 file_name = cdp.config_class_desc_csv,
                                                 header = None)


# Annotation Dataframe (this is massive...)
annotation_df = mf.read_gcs_csv_to_pandas(bucket_name = cdp.config_source_bucket_name,
                                          file_name = cdp.config_train_annotations_csv)



    

# cdp.config_processed_bucket_name


### Quicker Way to Retrieve Information for 1 Class
###############################################################################

# Retrieve information about one image class using a class from src.image_manipulation
# (this will take a second... might want to close some Chrome tabs)
image_retriever = imm.OpenCVImageClassRetriever(class_name = 'Loveseat')

img_array = image_retriever.get_class_image_array()
bbox_df = image_retriever.get_bounding_box_df()
class_desc_df = image_retriever.get_class_desc_df()



### Write a File to GCS
###############################################################################

# Notice the 'processed_files' subfolder
df = pd.DataFrame(data=[{1,2,3},{4,5,6}],columns=['a','b','c'])
mf.write_csv_to_gcs(dframe = df, bucket_name = cdp.config_source_bucket_name, file_name = 'processed_files/Bidet/test_gcs_csv_write.csv')































