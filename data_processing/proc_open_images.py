### Overview
###############################################################################
# Processing images for selected classes and writing files to Google Cloud



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
import tempfile
from tensorflow.keras.preprocessing.image import load_img
import time
from skimage.transform import resize
import tqdm

# Import Project Modules
import src.config_data_processing as cdp
import src.image_manipulation as imm
import src.misc_functions as mf



### Run Script
###############################################################################     
if __name__ == '__main__':
    for c in cdp.config_obj_detection_classes:
        mf.print_timestamp_message(f"Starting image processing and GCS file write for image class '{c}'")
        image_retriever = imm.OpenCVImageClassRetriever(class_name = c, use_local = True)
        image_retriever.resize_and_save_images()













        