### Object Classes to Use in Data Processing
###############################################################################
config_obj_detection_classes = ['Bathroom cabinet', 'Bathtub', 'Bed', 'Bench', 'Bidet',
                                'Billiard table', 'Cabinetry', 'Cat furniture', 'Ceiling fan',
                                'Chair', 'Chest of drawers', 'Closet', 'Coffee table',
                                'Computer keyboard', 'Computer monitor', 'Couch', 'Countertop',
                                'Cupboard', 'Curtain', 'Desk', 'Dishwasher', 'Drawer',
                                'Filing cabinet', 'Fireplace', 'Furniture', 'Gas stove',
                                'Heater', 'Houseplant', 'Infant bed', 'Jacuzzi',
                                'Kitchen & dining room table', 'Lamp', 'Laptop',
                                'Loveseat', 'Microwave oven', 'Mirror', 'Nightstand',
                                'Oven', 'Piano', 'Refrigerator', 'Sculpture', 'Shelf',
                                'Shower', 'Sink', 'Sofa bed', 'Sports equipment', 'Stairs',
                                'Stationary bicycle', 'Stool', 'Studio couch', 'Table', 
                                'Tablet computer', 'Telephone', 'Television', 'Toilet',
                                'Treadmill', 'Vase', 'Wall clock', 'Wardrobe',
                                'Washing machine', 'Wine rack', 'Wood-burning stove']


### File Paths ... TO DO: use relative path or shared location for data
###############################################################################
config_gcs_auth_json_path = 'C:/gcs_auth/iod_google_cred.json'
config_source_bucket_name = 'open_images_v6_source_files'
config_processed_bucket_subfolder = 'processed_files/'
config_train_image_csv = 'train-images-boxable-with-rotation.csv'
config_train_annotations_csv = 'train-annotations-human-imagelabels-boxable.csv'
config_train_bbox_csv = 'oidv6-train-annotations-bbox.csv'
config_class_desc_csv = 'class-descriptions-boxable.csv'


### Data Manipulation
###############################################################################
config_resize_height = 200
config_resize_width = 200





