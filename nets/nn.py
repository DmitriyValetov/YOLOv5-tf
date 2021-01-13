import math

import tensorflow as tf
from tensorflow import nn
from tensorflow.keras import layers

from utils import config

initializer = tf.random_normal_initializer(stddev=0.01)


def activation_fn(x):
    return nn.swish(x)


def conv(inputs, filters, kernel_size=1, strides=1):
    x = layers.Conv2D(filters, kernel_size, strides, 'same', use_bias=False, kernel_initializer=initializer)(inputs)
    x = layers.BatchNormalization(momentum=0.03)(x)
    x = layers.Activation(activation_fn)(x)
    return x


def residual(x, filters, add=True):
    inputs = x
    if add:
        x = conv(x, filters, 1)
        x = conv(x, filters, 3)
        x = inputs + x
    else:
        x = conv(x, filters, 1)
        x = conv(x, filters, 3)
    return x


def csp(x, filters, n, add=True):
    y = conv(x, filters // 2)
    for _ in range(n):
        y = residual(y, filters // 2, add)

    x = conv(x, filters // 2)
    x = layers.concatenate([x, y])

    x = conv(x, filters)
    return x


def build_model(training=True):
    depth = config.depth[config.versions.index(config.version)]
    width = config.width[config.versions.index(config.version)]

    inputs = layers.Input([config.image_size, config.image_size, 3])
    x = nn.space_to_depth(inputs, 2)
    x = conv(x, int(round(width * 64)), 3)
    x = conv(x, int(round(width * 128)), 3, 2)
    x = csp(x, int(round(width * 128)), int(round(depth * 3)))

    x = conv(x, int(round(width * 256)), 3, 2)
    x = csp(x, int(round(width * 256)), int(round(depth * 9)))
    x1 = x

    x = conv(x, int(round(width * 512)), 3, 2)
    x = csp(x, int(round(width * 512)), int(round(depth * 9)))
    x2 = x

    x = conv(x, int(round(width * 1024)), 3, 2)
    x = conv(x, int(round(width * 512)), 1, 1)
    x = layers.concatenate([x,
                            nn.max_pool(x, 5, 1, 'SAME'),
                            nn.max_pool(x, 9, 1, 'SAME'),
                            nn.max_pool(x, 13, 1, 'SAME')])
    x = conv(x, int(round(width * 1024)), 1, 1)
    x = csp(x, int(round(width * 1024)), int(round(depth * 3)), False)

    x = conv(x, int(round(width * 512)), 1)
    x3 = x
    x = layers.UpSampling2D()(x)
    x = layers.concatenate([x, x2])
    x = csp(x, int(round(width * 512)), int(round(depth * 3)), False)

    x = conv(x, int(round(width * 256)), 1)
    x4 = x
    x = layers.UpSampling2D()(x)
    x = layers.concatenate([x, x1])
    x = csp(x, int(round(width * 256)), int(round(depth * 3)), False)
    p3 = layers.Conv2D(3 * (len(config.class_dict) + 5), 1, name = f'p3_{len(config.class_dict)}', 
                       kernel_initializer=initializer)(x)

    x = conv(x, int(round(width * 256)), 3, 2)
    x = layers.concatenate([x, x4])
    x = csp(x, int(round(width * 512)), int(round(depth * 3)), False)
    p4 = layers.Conv2D(3 * (len(config.class_dict) + 5), 1, name = f'p4_{len(config.class_dict)}', 
                       kernel_initializer=initializer)(x)

    x = conv(x, int(round(width * 512)), 3, 2)
    x = layers.concatenate([x, x3])
    x = csp(x, int(round(width * 1024)), int(round(depth * 3)), False)
    p5 = layers.Conv2D(3 * (len(config.class_dict) + 5), 1, name = f'p5_{len(config.class_dict)}',
                       kernel_initializer=initializer)(x)

    if training:
        return tf.keras.Model(inputs, [p5, p4, p3])
    else:
        return tf.keras.Model(inputs, predict([p5, p4, p3]))


def process_layer(feature_map, anchors):
    grid_size = tf.shape(feature_map)[1:3]
    ratio = tf.cast(tf.constant([config.image_size, config.image_size]) / grid_size, tf.float32)
    rescaled_anchors = [(anchor[0] / ratio[1], anchor[1] / ratio[0]) for anchor in anchors]

    feature_map = tf.reshape(feature_map, [-1, grid_size[0], grid_size[1], 3, 5 + len(config.class_dict)])

    box_centers, box_sizes, conf, prob = tf.split(feature_map, [2, 2, 1, len(config.class_dict)], axis=-1)
    box_centers = tf.nn.sigmoid(box_centers)

    grid_x = tf.range(grid_size[1], dtype=tf.int32)
    grid_y = tf.range(grid_size[0], dtype=tf.int32)
    grid_x, grid_y = tf.meshgrid(grid_x, grid_y)
    x_offset = tf.reshape(grid_x, (-1, 1))
    y_offset = tf.reshape(grid_y, (-1, 1))
    x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
    x_y_offset = tf.cast(tf.reshape(x_y_offset, [grid_size[0], grid_size[1], 1, 2]), tf.float32)

    box_centers = box_centers + x_y_offset
    box_centers = box_centers * ratio[::-1]

    box_sizes = tf.exp(box_sizes) * rescaled_anchors
    box_sizes = box_sizes * ratio[::-1]

    boxes = tf.concat([box_centers, box_sizes], axis=-1)

    return x_y_offset, boxes, conf, prob


def reshape(features):
    x_y_offset, boxes, conf, prob = features
    grid_size = tf.shape(x_y_offset)[:2]
    boxes = tf.reshape(boxes, [-1, grid_size[0] * grid_size[1] * 3, 4])
    conf = tf.reshape(conf, [-1, grid_size[0] * grid_size[1] * 3, 1])
    prob = tf.reshape(prob, [-1, grid_size[0] * grid_size[1] * 3, len(config.class_dict)])
    return boxes, conf, prob


def predict(feature_maps):
    feature_map_1, feature_map_2, feature_map_3 = feature_maps

    feature_map_anchors = [(feature_map_1, config.anchors[6:9]),
                           (feature_map_2, config.anchors[3:6]),
                           (feature_map_3, config.anchors[0:3])]
    reorg_results = [process_layer(feature_map, anchors) for (feature_map, anchors) in feature_map_anchors]

    boxes_list, conf_list, prob_list = [], [], []
    for result in reorg_results:
        box, conf, prob = reshape(result)
        boxes_list.append(box)
        conf_list.append(tf.sigmoid(conf))
        prob_list.append(tf.sigmoid(prob))

    boxes = tf.concat(boxes_list, axis=1)
    conf = tf.concat(conf_list, axis=1)
    prob = tf.concat(prob_list, axis=1)

    center_x, center_y, width, height = tf.split(boxes, [1, 1, 1, 1], axis=-1)
    x_min = center_x - width / 2
    y_min = center_y - height / 2
    x_max = center_x + width / 2
    y_max = center_y + height / 2

    boxes = tf.concat([x_min, y_min, x_max, y_max], axis=-1)
    return gpu_nms(boxes, conf * prob, len(config.class_dict))


def gpu_nms(boxes, scores, num_classes, max_boxes=150, score_thresh=1e-5, nms_thresh=0.5):
    boxes_list, label_list, score_list = [], [], []
    max_boxes = tf.constant(max_boxes, dtype='int32')

    boxes = tf.reshape(boxes, [-1, 4])
    score = tf.reshape(scores, [-1, num_classes])

    mask = tf.greater_equal(score, tf.constant(score_thresh))
    for i in range(num_classes):
        filter_boxes = tf.boolean_mask(boxes, mask[:, i])
        filter_score = tf.boolean_mask(score[:, i], mask[:, i])
        nms_indices = tf.image.non_max_suppression(boxes=filter_boxes,
                                                   scores=filter_score,
                                                   max_output_size=max_boxes,
                                                   iou_threshold=nms_thresh, name='nms_indices')
        label_list.append(tf.ones_like(tf.gather(filter_score, nms_indices), 'int32') * i)
        boxes_list.append(tf.gather(filter_boxes, nms_indices))
        score_list.append(tf.gather(filter_score, nms_indices))

    boxes = tf.concat(boxes_list, axis=0)
    score = tf.concat(score_list, axis=0)
    label = tf.concat(label_list, axis=0)
    return boxes, score, label


def box_iou(pred_boxes, valid_true_boxes):
    pred_box_xy = pred_boxes[..., 0:2]
    pred_box_wh = pred_boxes[..., 2:4]

    pred_box_xy = tf.expand_dims(pred_box_xy, -2)
    pred_box_wh = tf.expand_dims(pred_box_wh, -2)

    true_box_xy = valid_true_boxes[:, 0:2]
    true_box_wh = valid_true_boxes[:, 2:4]

    intersect_min = tf.maximum(pred_box_xy - pred_box_wh / 2., true_box_xy - true_box_wh / 2.)
    intersect_max = tf.minimum(pred_box_xy + pred_box_wh / 2., true_box_xy + true_box_wh / 2.)

    intersect_wh = tf.maximum(intersect_max - intersect_min, 0.)

    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    pred_box_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]
    true_box_area = true_box_wh[..., 0] * true_box_wh[..., 1]
    true_box_area = tf.expand_dims(true_box_area, axis=0)

    iou = intersect_area / (pred_box_area + true_box_area - intersect_area + 1e-10)

    return iou


def process_loss(feature_map_i, y_true, anchors):
    grid_size = tf.shape(feature_map_i)[1:3]
    ratio = tf.cast(tf.constant([config.image_size, config.image_size]) / grid_size, tf.float32)
    batch_size = tf.cast(tf.shape(feature_map_i)[0], tf.float32)

    x_y_offset, pred_boxes, pred_conf, pred_prob = process_layer(feature_map_i, anchors)

    object_mask = y_true[..., 4:5]

    def loop_cond(idx, _):
        return tf.less(idx, tf.cast(batch_size, tf.int32))

    def loop_body(idx, mask):
        valid_true_boxes = tf.boolean_mask(y_true[idx, ..., 0:4], tf.cast(object_mask[idx, ..., 0], 'bool'))
        iou = box_iou(pred_boxes[idx], valid_true_boxes)
        best_iou = tf.reduce_max(iou, axis=-1)
        ignore_mask_tmp = tf.cast(best_iou < 0.5, tf.float32)
        mask = mask.write(idx, ignore_mask_tmp)
        return idx + 1, mask

    ignore_mask = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

    _, ignore_mask = tf.while_loop(cond=loop_cond, body=loop_body, loop_vars=[0, ignore_mask])
    ignore_mask = ignore_mask.stack()
    ignore_mask = tf.expand_dims(ignore_mask, -1)

    pred_box_xy = pred_boxes[..., 0:2]
    pred_box_wh = pred_boxes[..., 2:4]

    true_xy = y_true[..., 0:2] / ratio[::-1] - x_y_offset
    pred_xy = pred_box_xy / ratio[::-1] - x_y_offset

    true_tw_th = y_true[..., 2:4] / anchors
    pred_tw_th = pred_box_wh / anchors
    true_tw_th = tf.where(condition=tf.equal(true_tw_th, 0), x=tf.ones_like(true_tw_th), y=true_tw_th)
    pred_tw_th = tf.where(condition=tf.equal(pred_tw_th, 0), x=tf.ones_like(pred_tw_th), y=pred_tw_th)
    true_tw_th = tf.math.log(tf.clip_by_value(true_tw_th, 1e-9, 1e9))
    pred_tw_th = tf.math.log(tf.clip_by_value(pred_tw_th, 1e-9, 1e9))

    box_loss_scale_1 = y_true[..., 2:3] / tf.cast(tf.constant([config.image_size, config.image_size])[1], tf.float32)
    box_loss_scale_2 = y_true[..., 3:4] / tf.cast(tf.constant([config.image_size, config.image_size])[0], tf.float32)

    box_loss_scale = 2. - box_loss_scale_1 * box_loss_scale_2

    mix_w = y_true[..., -1:]
    xy_loss = tf.reduce_sum(tf.square(true_xy - pred_xy) * object_mask * box_loss_scale * mix_w)
    wh_loss = tf.reduce_sum(tf.square(true_tw_th - pred_tw_th) * object_mask * box_loss_scale * mix_w)

    conf_pos_mask = object_mask
    conf_neg_mask = (1 - object_mask) * ignore_mask
    conf_loss_pos = conf_pos_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=pred_conf)
    conf_loss_neg = conf_neg_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=pred_conf)
    conf_loss = conf_loss_pos + conf_loss_neg

    alpha = 0.25
    gamma = 1.5
    focal_mask = alpha * tf.pow(tf.abs(object_mask - tf.sigmoid(pred_conf)), gamma)
    conf_loss *= focal_mask

    conf_loss = tf.reduce_sum(conf_loss * mix_w)

    delta = 0.01
    label_target = (1 - delta) * y_true[..., 5:-1] + delta * 1. / len(config.class_dict)

    class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_target, logits=pred_prob) * mix_w
    class_loss = tf.reduce_sum(class_loss)

    return xy_loss, wh_loss, conf_loss, class_loss


def compute_loss(y_pred, y_true):
    loss_xy, loss_wh, loss_conf, loss_class = 0., 0., 0., 0.
    anchor_group = [config.anchors[6:9], config.anchors[3:6], config.anchors[0:3]]

    for i in range(len(y_pred)):
        result = process_loss(y_pred[i], y_true[i], anchor_group[i])
        loss_xy += result[0]
        loss_wh += result[1]
        loss_conf += result[2]
        loss_class += result[3]
    return loss_xy + loss_wh + loss_conf + loss_class


class CosineLrSchedule(tf.optimizers.schedules.LearningRateSchedule):
    def __init__(self, warmup_step):
        super().__init__()
        self.lr_min = 1e-4
        self.lr_max = 1e-3
        self.warmup_step = warmup_step
        self.decay_steps = tf.cast(warmup_step * (config.num_epochs - 1), tf.float32)

    def __call__(self, step):
        cos = tf.cos(math.pi * (tf.cast(step, tf.float32) - self.warmup_step) / self.decay_steps)
        linear_warmup = tf.cast(step, dtype=tf.float32) / self.warmup_step * self.lr_max
        cos_lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + cos / self.decay_steps)

        return tf.where(step < self.warmup_step, linear_warmup, cos_lr)

    def get_config(self):
        return {"lr_min": self.lr_min,
                "lr_max": self.lr_max,
                "decay_steps": self.decay_steps,
                "lr_warmup_step": self.warmup_step}
