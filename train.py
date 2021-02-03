import os

import tensorflow as tf

from data_reader import get_train_data, get_val_data
from utils import save_model, draw_train_history

from config.mbnv2_384x576_32x3_300_imgnet import MyNet

'''GPU Setting'''
os.environ['CUDA_VISIBLE_DEVICES'] = "7"
num_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
strategy = tf.distribute.MirroredStrategy()
net = MyNet(32 * num_gpus)

'''Get Dataset'''
train_ds = get_train_data(
    height=net.height,
    width=net.width,
    batch_size=net.batch_size)
val_ds = get_val_data(
    height=net.height,
    width=net.width,
    batch_size=1)

'''Build Model and set train'''
with strategy.scope():
    net.init_model()

'''Checkpoint Setting'''
checkpoint_path = "checkpoints/" + net.name
if not os.path.exists(checkpoint_path):
    os.mkdir(checkpoint_path)
checkpoint_prefix = os.path.join(checkpoint_path, "ckpt_{epoch}")

'''Start Train'''
log_dir = "./logs/"+net.name
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
history = net.model.fit(train_ds,
                        epochs=net.epochs,
                        shuffle=False,
                        validation_data=val_ds,
                        validation_steps=len(val_ds),
                        callbacks=[tf.keras.callbacks.TensorBoard(log_dir=log_dir),
                                   tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                                                      save_weights_only=True)],
                        verbose=1)

'''Save Last Model'''
save_model(model_name=net.name, model=net.model)

'''Draw Train History'''
draw_train_history(history)
