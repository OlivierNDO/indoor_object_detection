e### Overview
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
        image_retriever = imm.OpenCVImageClassRetriever(class_name = c, max_images = 5000)
        image_retriever.resize_and_save_images()
