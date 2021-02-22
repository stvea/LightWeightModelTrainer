import os
import re

import tensorflow as tf
from data_reader import get_train_data, get_val_data
from utills.args import get_train_checkpoint_args
import tensorflow_model_optimization as tfmot
import numpy as np

from utills.base import mkdir
from utills.converter import convert_from_keras_model

'''Set Start'''
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
train_name = "t2021_02_22_14_39_mb2_s_224"
epochs = 30
epochs_pruning = 10
'''Set End'''

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

NetConfig = None
exec('from save.' + train_name + '.net_config import *')

model_path = "save/" + train_name + "/checkpoint/"
prune_checkpoint_path = "save/" + train_name + "/prune_checkpoint/"
save_path = "save/" + train_name + "/"
mkdir(prune_checkpoint_path)
train_generator = get_train_data()
val_generator = get_val_data()

model = tf.keras.models.load_model(model_path)
losses = tf.losses.CategoricalCrossentropy(label_smoothing=NetConfig.LABEL_SMOOTHING)

end_step = np.ceil(345).astype(np.int32) * epochs_pruning

pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.5,
                                                        final_sparsity=0.80,
                                                        begin_step=0,
                                                        end_step=end_step)

pruned_model = tf.keras.Sequential()
for layer in model.layers:
    try:
        pruned_model.add(prune_low_magnitude(
            layer,
            pruning_schedule,
            block_size=(1, 1)
        ))
    except:
        pruned_model.add(layer)

pruned_model.compile(
    loss=losses,
    optimizer=tf.keras.optimizers.Adam(1e-5),
    metrics=['accuracy'])

print(pruned_model.layers[0].summary())
print(pruned_model.summary())

history_fine = pruned_model.fit(train_generator,
                                steps_per_epoch=len(train_generator),
                                epochs=epochs,
                                validation_data=val_generator,
                                validation_freq=5,
                                callbacks=[
                                    tf.keras.callbacks.ModelCheckpoint(
                                        filepath=prune_checkpoint_path,
                                        monitor='val_accuracy',
                                        save_best_only=1,
                                        verbose=1),
                                    tfmot.sparsity.keras.UpdatePruningStep()
                                ])

model_for_export = tfmot.sparsity.keras.strip_pruning(pruned_model)

convert_from_keras_model(model_for_export, _image_size=NetConfig.IMAGE_SIZE,
                         tflite_save_path=save_path + "purning_model.tflite")

print('Saved pruned TFLite model to:', save_path + "purning_model.tflite")
