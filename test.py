import os
import pickle
from os.path import exists
from os.path import join
from xml.etree.ElementTree import parse as parse_fn

import cv2
import numpy as np
import tqdm

from nets import nn
from utils import config
from utils import util
from utils.util import parse_annotation


def draw_bbox(image, boxes, scores):
    for box, score in zip(boxes, scores):
        if score > 0.2:
            coordinate = np.array(box[:4], dtype=np.int32)
            c1, c2 = (coordinate[0], coordinate[1]), (coordinate[2], coordinate[3])
            cv2.rectangle(image, c1, c2, (255, 0, 0), 1)
    return image


def main():
    if not exists('results'):
        os.makedirs('results')
    f_names = []
    with open(join(config.base_dir, 'val.txt')) as reader:
        lines = reader.readlines()
    for line in lines:
        f_names.append(line.rstrip().split(' ')[0])
    result_dict = {}

    model = nn.build_model(training=False)
    model.load_weights("weights/model245.h5", True)

    for f_name in tqdm.tqdm(f_names):
        image_path = join(config.base_dir, config.image_dir, f_name + '.jpg')
        label_path = join(config.base_dir, config.label_dir, f_name + '.xml')
        image = cv2.imread(image_path)
        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_np, dw, dh, scale = util.resize(image_np)
        image_np = image_np.astype(np.float32) / 255.0

        boxes, scores, _ = model.predict(image_np[np.newaxis, ...])
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / scale
        true_boxes = []
        for element in parse_fn(label_path).getroot().iter('object'):
            true_boxes.append(parse_annotation(element)[2])
        result = {'pred_boxes': boxes,
                  'true_boxes': true_boxes,
                  'confidence': scores}
        result_dict[f'{f_name}.jpg'] = result
        image = draw_bbox(image, boxes, scores)
        cv2.imwrite(f'results/{f_name}.png', image)
    with open(join('results/yolo.pickle'), 'wb') as writer:
        pickle.dump(result_dict, writer)


if __name__ == '__main__':
    main()
