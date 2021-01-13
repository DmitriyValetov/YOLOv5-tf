import os
import sys
from os.path import exists
from os.path import join

import numpy as np
import tensorflow as tf
from utils import config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

np.random.seed(12345)
tf.random.set_seed(12345)

from nets import nn
from utils import data_loader

strategy = tf.distribute.MirroredStrategy()

file_names = []
with open(join(config.base_dir, 'train.txt')) as reader:
    for line in reader.readlines():
        image_path = join(config.base_dir, config.image_dir, line.rstrip().split(' ')[0]+'.jpg')
        label_path = join(config.base_dir, config.label_dir, line.rstrip().split(' ')[0]+'.xml')
        if exists(image_path) and exists(label_path):
            file_names.append(join(config.base_dir, 'TF', line.rstrip().split(' ')[0] + '.tf'))
print(f'[INFO] {len(file_names)} data points')
num_replicas = strategy.num_replicas_in_sync
steps = len(file_names) // config.batch_size
lr = nn.CosineLrSchedule(steps)

dataset = data_loader.DataLoader().input_fn(file_names)
dataset = strategy.experimental_distribute_dataset(dataset)

with strategy.scope():
    model = nn.build_model()
    model.summary()
    optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.935, decay=0.0005)

with strategy.scope():
    loss_object = nn.compute_loss


    def compute_loss(y_true, y_pred):
        total_loss = loss_object(y_pred, y_true)
        return tf.reduce_sum(total_loss) / config.batch_size


def train_step(image, y_true):
    with tf.GradientTape() as tape:
        y_pred = model(image, training=True)
        loss = compute_loss(y_true, y_pred)
    variables = model.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return loss


@tf.function
def distributed_train_step(image, y_true):
    per_replica_losses = strategy.run(train_step, args=(image, y_true))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)


def main():
    if not exists('weights'):
        os.makedirs('weights')
    pb = tf.keras.utils.Progbar(steps, stateful_metrics=['loss'])
    for step, inputs in enumerate(dataset):
        if step % steps == 0:
            print(f'Epoch {step // steps + 1}/{config.num_epochs}')
            pb = tf.keras.utils.Progbar(steps, stateful_metrics=['loss'])
        step += 1
        image, y_true_1, y_true_2, y_true_3 = inputs
        y_true = (y_true_1, y_true_2, y_true_3)
        loss = distributed_train_step(image, y_true)
        pb.add(1, [('loss', loss)])
        if step % steps == 0:
            model.save_weights(join("weights", f"model_{config.version}.h5"))
        if step // steps == config.num_epochs:
            sys.exit("--- Stop Training ---")


if __name__ == '__main__':
    main()
