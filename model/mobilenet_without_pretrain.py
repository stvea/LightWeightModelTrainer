import os

import tensorflow as tf

from data_reader import *

os.environ['CUDA_VISIBLE_DEVICES'] = "5,6,7"

# strategy = tf.distribute.MirroredStrategy()
# train_ds = get_train_data()
# val_ds = get_val_data()
# loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
#
# with strategy.scope():
#     model = tf.keras.applications.MobileNetV2(
#         input_shape=(384, 576, 3), weights=None, classes=30,
#         classifier_activation='relu'
#     )
#     model.compile(optimizer=optimizer, loss=loss_object, metrics=['accuracy',tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3)])
# # checkpoint_dir = './training_checkpoints'
# # if not os.path.exists(checkpoint_dir):
# #     os.mkdir(checkpoint_dir)
# # checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
# model.fit(train_ds,
#           epochs=10,
#           shuffle=False,
#           validation_data=val_ds,
#           validation_freq=5,
#           # validation_steps=3,
#           verbose=1)
#
#
saved_model_dir = 'saved_model/test'
# if not os.path.exists(saved_model_dir):
#     os.mkdir(saved_model_dir)
#
# tf.saved_model.save(model, saved_model_dir)
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
