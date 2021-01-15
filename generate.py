import multiprocessing
import os
from multiprocessing import Process
from multiprocessing import cpu_count
from os.path import exists
from os.path import join

import numpy as np
import tensorflow as tf
import tqdm

from utils import config
from utils import util


class AnchorGenerator:
    def __init__(self, cluster_number):
        self.cluster_number = cluster_number

    def iou(self, boxes, clusters):  # 1 box -> k clusters
        n = boxes.shape[0]
        k = self.cluster_number

        box_area = boxes[:, 0] * boxes[:, 1]
        box_area = box_area.repeat(k)
        box_area = np.reshape(box_area, (n, k))

        cluster_area = clusters[:, 0] * clusters[:, 1]
        cluster_area = np.tile(cluster_area, [1, n])
        cluster_area = np.reshape(cluster_area, (n, k))

        box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
        cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
        min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

        box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
        cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
        min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
        inter_area = np.multiply(min_w_matrix, min_h_matrix)

        result = inter_area / (box_area + cluster_area - inter_area)
        return result

    def avg_iou(self, boxes, clusters):
        accuracy = np.mean([np.max(self.iou(boxes, clusters), axis=1)])
        return accuracy

    def generator(self, boxes, k, dist=np.median):
        box_number = boxes.shape[0]
        last_nearest = np.zeros((box_number,))
        clusters = boxes[np.random.choice(box_number, k, replace=False)]  # init k clusters
        while True:
            distances = 1 - self.iou(boxes, clusters)

            current_nearest = np.argmin(distances, axis=1)
            if (last_nearest == current_nearest).all():
                break  # clusters won't change
            for cluster in range(k):
                clusters[cluster] = dist(boxes[current_nearest == cluster], axis=0)
            last_nearest = current_nearest

        return clusters

    def generate_anchor(self):
        boxes = self.get_boxes()
        result = self.generator(boxes, k=self.cluster_number)
        result = result[np.lexsort(result.T[0, None])]
        print("K anchors:\n {}".format(result))
        print("Accuracy: {:.2f}%".format(self.avg_iou(boxes, result) * 100))

    @staticmethod
    def get_boxes():
        boxes = []
        file_names = [file_name[:-4] for file_name in os.listdir(join(config.base_dir, config.label_dir))]
        for file_name in file_names:
            for annotation in util.load_label(file_name)[0]:
                x_min = annotation[0]
                y_min = annotation[1]
                x_max = annotation[2]
                y_max = annotation[3]
                width = x_max - x_min
                height = y_max - y_min
                boxes.append([width, height])
        return np.array(boxes)


def byte_feature(value):
    if not isinstance(value, bytes):
        if not isinstance(value, list):
            value = value.encode('utf-8')
        else:
            value = [val.encode('utf-8') for val in value]
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def build_example(file_name):
    in_image = util.load_image(file_name)
    boxes, label = util.load_label(file_name)
    boxes = np.concatenate((boxes, np.full(shape=(boxes.shape[0], 1), fill_value=1., dtype=np.float32)), axis=-1)

    in_image, boxes = util.resize(in_image, boxes)

    y_true_1, y_true_2, y_true_3 = util.process_box(boxes, label)

    in_image = in_image.astype('float32')
    y_true_1 = y_true_1.astype('float32')
    y_true_2 = y_true_2.astype('float32')
    y_true_3 = y_true_3.astype('float32')

    in_image = in_image.tobytes()
    y_true_1 = y_true_1.tobytes()
    y_true_2 = y_true_2.tobytes()
    y_true_3 = y_true_3.tobytes()

    features = tf.train.Features(feature={'in_image': byte_feature(in_image),
                                          'y_true_1': byte_feature(y_true_1),
                                          'y_true_2': byte_feature(y_true_2),
                                          'y_true_3': byte_feature(y_true_3)})

    return tf.train.Example(features=features)


def write_tf_record(queue, sentinel):
    while True:
        file_name = queue.get()

        if file_name == sentinel:
            break
        tf_example = build_example(file_name)
        opt = tf.io.TFRecordOptions('GZIP')
        with tf.io.TFRecordWriter(join(config.base_dir, 'TF', file_name + ".tf"), opt) as writer:
            writer.write(tf_example.SerializeToString())


def generate_tf_record():
    if not exists(join(config.base_dir, 'TF')):
        os.makedirs(join(config.base_dir, 'TF'))
    file_names = []
    with open(join(config.base_dir, 'train.txt')) as reader:
        for line in reader.readlines():
            file_names.append(line.rstrip().split(' ')[0])
    sentinel = ("", [])
    queue = multiprocessing.Manager().Queue()
    for file_name in tqdm.tqdm(file_names):
        queue.put(file_name)
    for _ in range(cpu_count()):
        queue.put(sentinel)
    print('[INFO] generating TF record')
    process_pool = []
    for i in range(cpu_count()):
        process = Process(target=write_tf_record, args=(queue, sentinel))
        process_pool.append(process)
        process.start()
    for process in process_pool:
        process.join()


if __name__ == "__main__":
    generator = AnchorGenerator(9)
    generator.generate_anchor()
    generate_tf_record()
