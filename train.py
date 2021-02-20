from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import time
import tensorflow as tf

from net_config import *
from data_reader import get_train_data, get_val_data
from model.core import build_model
from utills.args import get_train_args
from utills.converter import convert_from_save_model
from utills.draw_train_plot import draw_train_history
from utills.base import init_train
from shutil import copyfile

args = get_train_args()

model_name = "mb2"
train_name = \
    "t" + str(time.strftime("%Y_%m_%d_%H_%M", time.localtime())) + \
    "_" + model_name + "_" + str(NetConfig.IMAGE_SIZE)

os.environ['CUDA_VISIBLE_DEVICES'] = NetConfig.GPU

train_generator = get_train_data()
val_generator = get_val_data()

save_path = "save/" + train_name
val_data_path = 'data/val_new/'
checkpoint_path = "save/" + train_name + "/checkpoint"

log = init_train(train_name,save_path,checkpoint_path)

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = build_model(model_name,NetConfig.IMG_SHAPE,strategy)
    base_model = model.layers[0]

    losses = tf.losses.CategoricalCrossentropy(label_smoothing=NetConfig.LABEL_SMOOTHING)

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=losses,
                  metrics=['accuracy'])

    history = model.fit(train_generator,
                        steps_per_epoch=len(train_generator),
                        epochs=NetConfig.EPOCHS_TRAIN,
                        validation_data=val_generator,
                        validation_freq=5,
                        callbacks=[
                            tf.keras.callbacks.ModelCheckpoint(
                                filepath=checkpoint_path,
                                monitor='val_accuracy',
                                save_best_only=1,
                                verbose=1)
                        ])
    # draw_train_history(history)

    base_model.trainable = True
    for layer in base_model.layers[:NetConfig.FINE_TUNE_START]:
        layer.trainable = False

    model.compile(
        loss=losses,
        optimizer=tf.keras.optimizers.Adam(1e-5),
        metrics=['accuracy'])

    history_fine = model.fit(train_generator,
                             steps_per_epoch=len(train_generator),
                             epochs=NetConfig.EPOCHS_FINE,
                             validation_data=val_generator,
                             validation_freq=5,
                             callbacks=[
                                 tf.keras.callbacks.ModelCheckpoint(
                                     filepath=checkpoint_path,
                                     monitor='val_accuracy',
                                     save_best_only=1,
                                     verbose=1)
                             ])

    model.layers[2].rate = 0
    results = model.evaluate(val_generator)

    log.info("set dropout = 0")
    log.info("test_model loss, test_model acc:", results)

    tf.saved_model.save(model, save_path)

    convert_from_save_model(save_path, tflite_save_path=save_path + "/model.tflite", type="dynamic")
    draw_train_history(history)
