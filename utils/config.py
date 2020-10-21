from os.path import join
import numpy as np

epochs = 300
batch_size = 18
image_size = 1024
base_dir = join('..', 'Dataset', 'Dubai')
image_dir = 'IMAGES'
label_dir = 'LABELS'
classes = {'DAMAGE': 0}
strides = [8, 16, 32]
anchors = np.array([[9, 8], [12, 11], [14, 15],
                    [17, 8], [18, 24], [25, 13],
                    [35, 29], [46, 13], [91, 24]], np.float32)
