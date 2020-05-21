### Object Classes to Use in Data Processing
###############################################################################
config_obj_detection_classes = ['Bathtub', 'Countertop', 'Cabinetry','Ceiling fan',
                                'Coffee table', 'Couch', 'Desk', 'Dishwasher', 'Fireplace',
                                'Furniture', 'Heater', 'Jacuzzi', 'Kitchen & dining room table',
                                'Loveseat', 'Microwave oven', 'Nightstand', 'Oven', 'Piano',
                                'Refrigerator', 'Table', 'Television', 'Toilet',
                                'Training bench', 'Treadmill', 'Washing machine']


### File Paths ... TO DO: use relative path or shared location for data
###############################################################################
raw_data_dir = 'D:/indoor_object_detection/data/source_data/'

class_desc_file = f'{raw_data_dir}class-descriptions-boxable.csv'
train_bbox_file = f'{raw_data_dir}oidv6-train-annotations-bbox.csv'
test_bbox_file = f'{raw_data_dir}test-annotations-bbox.csv'
valid_bbox_file = f'{raw_data_dir}validation-annotations-bbox.csv'

train_label_file = f'{raw_data_dir}train-annotations-human-imagelabels-boxable.csv'
test_label_file = f'{raw_data_dir}test-annotations-human-imagelabels-boxable.csv'
valid_label_file = f'{raw_data_dir}validation-annotations-human-imagelabels-boxable.csv'

train_image_id_file = 'train-images-boxable-with-rotation.csv'
test_image_id_file = 'test-images-boxable-with-rotation.csv'
valid_image_id_file = 'test-images-boxable-with-rotation.csv'
