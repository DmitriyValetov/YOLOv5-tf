import os
import sys
from os.path import exists
from os.path import join

import numpy as np
import tensorflow as tf

from nets import nn
from utils import config
from utils.dataset import DataLoader

np.random.seed(12345)
tf.random.set_seed(12345)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
strategy = tf.distribute.MirroredStrategy()

file_names = []
with open(join(config.base_dir, 'train.txt')) as reader:
    for line in reader.readlines():
        image_path = join(config.base_dir, config.image_dir, line.rstrip() + '.jpg')
        label_path = join(config.base_dir, config.label_dir, line.rstrip() + '.xml')
        if exists(image_path) and exists(label_path):
            file_names.append(line.rstrip())
print(f'[INFO] {len(file_names)} data points')
num_replicas = strategy.num_replicas_in_sync
steps = len(file_names) // config.batch_size

dataset = DataLoader().input_fn(file_names)
dataset = strategy.experimental_distribute_dataset(dataset)

with strategy.scope():
    model = nn.build_model()
    model.summary()
    optimizer = tf.keras.optimizers.Adam(nn.CosineLrSchedule(steps), 0.937)

with strategy.scope():
    loss_object = nn.ComputeLoss()


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
