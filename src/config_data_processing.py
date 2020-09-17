### Object Classes to Use in Data Processing
###############################################################################
config_obj_detection_classes = ['Sports equipment', 'Chair', 'Sculpture', 'Laptop', 'Desk', 'Houseplant',
                                'Shelf', 'Couch', 'Stairs', 'Vase', 'Bench', 'Computer monitor',
                                'Computer keyboard', 'Sink', 'Bed', 'Cabinetry', 'Coffee table', 
                                'Television', 'Curtain', 'Piano', 'Mirror', 'Countertop',
                                'Kitchen & dining room table', 'Drawer', 'Lamp', 'Fireplace',
                                'Studio couch', 'Chest of drawers', 'Sofa bed']


### File Paths ... TO DO: use relative path or shared location for data
###############################################################################
config_gcs_auth_json_path = 'C:/gcs_auth/iod_google_cred.json'
config_source_bucket_name = 'open_images_v6_source_files'
config_processed_bucket_subfolder = 'processed_files/'
config_train_image_csv = 'train-images-boxable-with-rotation.csv'
config_train_annotations_csv = 'train-annotations-human-imagelabels-boxable.csv'
config_train_bbox_csv = 'oidv6-train-annotations-bbox.csv'
config_class_desc_csv = 'class-descriptions-boxable.csv'
config_temp_local_folder = 'D:/numpy_transfer/'

### Data Manipulation
###############################################################################
config_resize_height = 200
config_resize_width = 200





