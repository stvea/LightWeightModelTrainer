import os
import tensorflow as tf
from data_reader import get_train_data, get_val_data
from utills.args import get_train_checkpoint_args


os.environ['CUDA_VISIBLE_DEVICES'] = "6"
train_name = "t2021_02_20_11_15_mb2_384"
NetConfig = None
exec('from save.' + train_name + '.net_config import *')

model_path = "save/" + train_name + "/checkpoint/"
checkpoint_path = "save/" + train_name + "/checkpoint/"
save_path = "save/" + train_name

train_generator = get_train_data()
val_generator = get_val_data()

model = tf.keras.models.load_model(model_path)
losses = tf.losses.CategoricalCrossentropy(label_smoothing=NetConfig.LABEL_SMOOTHING)

model.layers[0].trainable = True
for layer in model.layers[0].layers[:NetConfig.FINE_TUNE_START]:
    layer.trainable = False

model.compile(
    loss=losses,
    optimizer=tf.keras.optimizers.Adam(2e-5),
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
                                 verbose=1)])