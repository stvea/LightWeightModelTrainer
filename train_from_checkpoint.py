import os
import tensorflow as tf

from data_reader import get_train_data, get_val_data

train_name = "ti2021_02_19_10_01_mb2_224_t5_f300_nper_b64_fs25_ls"
exec('from save.' + train_name + '.constant import*')

os.environ['CUDA_VISIBLE_DEVICES'] = "7"

model_path = "save/" + train_name + "/log_train/"

'''Get Data'''
train_generator = get_train_data()
val_generator = get_val_data()

model = tf.keras.models.load_model(model_path)
print(model.layers[0])
losses = tf.losses.CategoricalCrossentropy(label_smoothing=0.01)

checkpoint_path = "save/" + train_name + "/checkpoint"

model.layers[0].trainable = True
for layer in model.layers[0].layers[:FINE_TUNE_START]:
    layer.trainable = False

model.compile(
    loss=losses,
    optimizer=tf.keras.optimizers.Adam(9e-6),
    metrics=['accuracy'])
# layer.trainable = False
save_path = "save/" + train_name

train_log_path = os.path.join(save_path, "log_train")

history_fine = model.fit(train_generator,
                         steps_per_epoch=len(train_generator),
                         epochs=EPOCHS_FINE,
                         validation_data=val_generator,
                         validation_freq=5,
                         callbacks=[
                             tf.keras.callbacks.ModelCheckpoint(
                                 filepath=train_log_path,
                                 monitor='val_accuracy',
                                 save_best_only=1,
                                 verbose=1)])
