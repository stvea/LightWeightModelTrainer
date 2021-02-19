from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
import tensorflow as tf

from net_config import *
from data_reader import get_train_data, get_val_data
from utills.converter import convert_from_save_model
from utills.draw_train_plot import draw_train_history
from utills.base import mkdir
from shutil import copyfile

'''Device Setting'''
os.environ['CUDA_VISIBLE_DEVICES'] = GPU

'''General Setting'''
train_name = \
    "t" + str(time.strftime("%Y_%m_%d_%H_%M", time.localtime())) + \
    "_mb2_" + str(IMAGE_SIZE) + \
    "_t" + str(EPOCHS_TRAIN) + \
    "_f" + str(EPOCHS_FINE) + \
    "_n" + NORMALIZATION + \
    "_b" + str(BATCH_SIZE) + \
    "_fs" + str(FINE_TUNE_START)

save_path = "save/" + train_name
val_data_path = 'data/val_new/'
checkpoint_path = "save/" + train_name + "/s1_checkpoint"

print(train_name)
mkdir(save_path)
mkdir(checkpoint_path)

copyfile("net_config.py", os.path.join(save_path, "net_config.py"))

'''Get Data'''
train_generator = get_train_data()
val_generator = get_val_data()

'''Build Model'''
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   alpha=1.4,
                                                   weights='imagenet')
    base_model.trainable = False
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Conv2D(192, 3, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(30, activation='softmax')
    ])
    losses = tf.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING)
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=losses,
                  metrics=['accuracy'])

    '''Part1 Train classification net'''
    history = model.fit(train_generator,
                        steps_per_epoch=len(train_generator),
                        epochs=EPOCHS_TRAIN,
                        validation_data=val_generator,
                        validation_freq=5,
                        callbacks=[
                            tf.keras.callbacks.ModelCheckpoint(
                                filepath=checkpoint_path,
                                monitor='val_accuracy',
                                save_best_only=1,
                                verbose=1)
                        ])
    draw_train_history(history)

    '''Part2 fine tuning'''
    base_model.trainable = True

    for layer in base_model.layers[:FINE_TUNE_START]:
        layer.trainable = False

    model.compile(
        loss=losses,
        optimizer=tf.keras.optimizers.Adam(1e-5),
        metrics=['accuracy'])

    print(model.summary())

    '''Training'''

    history_fine = model.fit(train_generator,
                             steps_per_epoch=len(train_generator),
                             epochs=EPOCHS_FINE,
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
    print("set dropout = 0")
    results = model.evaluate(val_generator)
    print("test_model loss, test_model acc:", results)

    '''Part3 Convert to TFLite'''
    tf.saved_model.save(model, save_path)
    convert_from_save_model(save_path, tflite_save_path=save_path + "/model.tflite", type="dynamic")
    # converter = tf.lite.TFLiteConverter.from_saved_model(save_path)
    # tflite_model = converter.convert()
    #
    # with open(save_path + '/' + train_name + '.tflite', 'wb') as f:
    #     f.write(tflite_model)
    draw_train_history(history)
