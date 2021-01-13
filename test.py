import os
from os.path import exists
from os.path import join

import cv2
import numpy as np
import tqdm

from nets import nn
from utils import config
from utils import util

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def draw_bbox(image, boxes, scores):
    for box, score in zip(boxes, scores):
        if score > 0.3:
            coordinate = np.array(box[:4], dtype=np.int32)
            c1, c2 = (coordinate[0], coordinate[1]), (coordinate[2], coordinate[3])
            cv2.rectangle(image, c1, c2, (255, 0, 0), 1)
    return image


def main():
    if not exists('results'):
        os.makedirs('results')
    file_names = []
    with open(join(config.base_dir, 'test.txt')) as reader:
        lines = reader.readlines()
    for line in lines:
        file_names.append(line.rstrip().split(' ')[0])

    model = nn.build_model(training=False)
    model.load_weights(f"weights/model_{config.version}.h5", True)

    for file_name in tqdm.tqdm(file_names):
        image = cv2.imread(join(config.base_dir, config.image_dir, file_name + '.jpg'))
        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_np, scale, dw, dh = util.resize(image_np)
        image_np = image_np.astype(np.float32) / 255.0

        boxes, scores, _ = model.predict(image_np[np.newaxis, ...])
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / scale
        image = draw_bbox(image, boxes, scores)
        cv2.imwrite(f'results/{file_name}.jpg', image)


if __name__ == '__main__':
    main()
