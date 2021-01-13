from os.path import join
from xml.etree.ElementTree import ParseError
from xml.etree.ElementTree import parse as parse_fn

import cv2
import numpy as np
from six import raise_from

from utils import config


def find_node(parent, name, debug_name=None, parse=None):
    if debug_name is None:
        debug_name = name

    result = parent.find(name)
    if result is None:
        raise ValueError(f'missing element \'{debug_name}\'')
    if parse is not None:
        try:
            return parse(result.text)
        except ValueError as e:
            raise_from(ValueError(f'illegal value for \'{debug_name}\': {e}'), None)
    return result


def parse_annotation(element):
    truncated = find_node(element, 'truncated', parse=int)
    difficult = find_node(element, 'difficult', parse=int)

    class_name = find_node(element, 'name').text
    if class_name not in config.class_dict:
        raise ValueError(f'class name \'{class_name}\' not found in classes: {list(config.class_dict.keys())}')

    label = config.class_dict[class_name]

    box = find_node(element, 'bndbox')
    x_min = find_node(box, 'xmin', 'bndbox.xmin', parse=int)
    y_min = find_node(box, 'ymin', 'bndbox.ymin', parse=int)
    x_max = find_node(box, 'xmax', 'bndbox.xmax', parse=int)
    y_max = find_node(box, 'ymax', 'bndbox.ymax', parse=int)

    return truncated, difficult, [x_min, y_min, x_max, y_max], label


def parse_annotations(xml_root):
    boxes = []
    labels = []
    for i, element in enumerate(xml_root.iter('object')):
        truncated, difficult, box, label = parse_annotation(element)
        boxes.append(box)
        labels.append(label)

    boxes = np.asarray(boxes, np.float32)
    labels = np.asarray(labels, np.int32)
    return boxes, labels


def load_image(f_name):
    path = join(config.base_dir, config.image_dir, f_name + '.jpg')
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def load_label(f_name):
    try:
        tree = parse_fn(join(config.base_dir, config.label_dir, f_name + '.xml'))
        return parse_annotations(tree.getroot())
    except ParseError as error:
        raise_from(ValueError(f'invalid annotations file: {f_name}: {error}'), None)
    except ValueError as error:
        raise_from(ValueError(f'invalid annotations file: {f_name}: {error}'), None)


def resize(image, boxes=None):
    h, w, _ = image.shape

    scale = min(config.image_size / w, config.image_size / h)
    w = int(scale * w)
    h = int(scale * h)

    image_resized = cv2.resize(image, (w, h))

    image_padded = np.zeros(shape=[config.image_size, config.image_size, 3], dtype=np.uint8)
    dw, dh = (config.image_size - w) // 2, (config.image_size - h) // 2
    image_padded[dh:h + dh, dw:w + dw, :] = image_resized.copy()

    if boxes is None:
        return image_padded, scale, dw, dh

    else:
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + dw
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + dh

        return image_padded, boxes


def process_box(boxes, labels):
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    anchors = config.anchors
    box_centers = (boxes[:, 0:2] + boxes[:, 2:4]) / 2
    box_size = boxes[:, 2:4] - boxes[:, 0:2]

    y_true_1 = np.zeros((config.image_size // 32, config.image_size // 32, 3, 6 + len(config.class_dict)), np.float32)
    y_true_2 = np.zeros((config.image_size // 16, config.image_size // 16, 3, 6 + len(config.class_dict)), np.float32)
    y_true_3 = np.zeros((config.image_size // 8, config.image_size // 8, 3, 6 + len(config.class_dict)), np.float32)

    y_true_1[..., -1] = 1.
    y_true_2[..., -1] = 1.
    y_true_3[..., -1] = 1.

    y_true = [y_true_1, y_true_2, y_true_3]

    box_size = np.expand_dims(box_size, 1)

    min_np = np.maximum(- box_size / 2, - anchors / 2)
    max_np = np.minimum(box_size / 2, anchors / 2)

    whs = max_np - min_np

    overlap = whs[:, :, 0] * whs[:, :, 1]
    union = box_size[:, :, 0] * box_size[:, :, 1] + anchors[:, 0] * anchors[:, 1] - whs[:, :, 0] * whs[:, :, 1] + 1e-10

    iou = overlap / union
    best_match_idx = np.argmax(iou, axis=1)

    ratio_dict = {1.: 8., 2.: 16., 3.: 32.}
    for i, idx in enumerate(best_match_idx):
        feature_map_group = 2 - idx // 3
        ratio = ratio_dict[np.ceil((idx + 1) / 3.)]
        x = int(np.floor(box_centers[i, 0] / ratio))
        y = int(np.floor(box_centers[i, 1] / ratio))
        k = anchors_mask[feature_map_group].index(idx)
        c = labels[i]

        y_true[feature_map_group][y, x, k, :2] = box_centers[i]
        y_true[feature_map_group][y, x, k, 2:4] = box_size[i]
        y_true[feature_map_group][y, x, k, 4] = 1.
        y_true[feature_map_group][y, x, k, 5 + c] = 1.
        y_true[feature_map_group][y, x, k, -1] = boxes[i, -1]

    return y_true_1, y_true_2, y_true_3
