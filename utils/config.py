from os.path import join
import numpy as np

width = [0.5, 0.75, 1.0, 1.25]
depth = [0.33, 0.67, 1.0, 1.33]
versions = ['s', 'm', 'l', 'x']
max_boxes = 150
use_focal = False
use_smooth = False
score_threshold = 0.25

num_epochs = 300
batch_size = 90
image_size = 640
class_dict = {'person': 0, 'bicycle': 1, 'car': 2, 'motorcycle': 3, 'airplane': 4, 'bus': 5, 'train': 6, 'truck': 7,
              'boat': 8, 'traffic light': 9, 'fire': 10, 'hydrant': 11, 'stop sign': 12, 'parking meter': 13,
              'bench': 14, 'bird': 15, 'cat': 16, 'dog': 17, 'horse': 18, 'sheep': 19, 'cow': 20, 'elephant': 21,
              'bear': 22, 'zebra': 23, 'giraffe': 24, 'backpack': 25, 'umbrella': 26, 'handbag': 27, 'tie': 28,
              'suitcase': 29, 'frisbee': 30, 'skis': 31, 'snowboard': 32, 'sports ball': 33, 'kite': 34,
              'baseball bat': 35, 'baseball glove': 36, 'skateboard': 37, 'surfboard': 38, 'tennis racket': 39,
              'bottle': 40, 'wine glass': 41, 'cup': 42, 'fork': 43, 'knife': 44, 'spoon': 45, 'bowl': 46, 'banana': 47,
              'apple': 48, 'sandwich': 49, 'orange': 50, 'broccoli': 51, 'carrot': 52, 'hot dog': 53, 'pizza': 54,
              'donut': 55, 'cake': 56, 'chair': 57, 'couch': 58, 'potted plant': 59, 'bed': 60, 'dining table': 61,
              'toilet': 62, 'tv': 63, 'laptop': 64, 'mouse': 65, 'remote': 66, 'keyboard': 67, 'cell phone': 68,
              'microwave oven': 69, 'toaster': 70, 'sink': 71, 'refrigerator': 72, 'book': 73, 'clock': 74, 'vase': 75,
              'scissors': 76, 'teddy bear': 77, 'hair drier': 78, 'toothbrush': 79}
base_dir = join('..', 'Dataset', 'COCO')
image_dir = 'IMAGES'
label_dir = 'LABELS'
version = 's'
anchors = np.array([[8., 9.], [16., 24.], [28., 58.],
                    [41., 25.], [58., 125.], [71., 52.],
                    [129., 97.], [163., 218.], [384., 347.]], np.float32)
