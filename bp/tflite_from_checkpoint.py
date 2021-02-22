import os
import tensorflow as tf

from data_reader import get_train_data, get_val_data

train_name = "ti2021_02_19_10_01_mb2_224_t5_f300_nper_b64_fs25_ls0.1"
# exec('from save.' + train_name + '.constant import*')
save_path = "save/" + train_name

os.environ['CUDA_VISIBLE_DEVICES'] = "6"
model_path = "save/" + train_name + "/log_train/"

'''Get Data'''
train_generator = get_train_data()
val_generator = get_val_data()

model = tf.keras.models.load_model(model_path)
print(model.layers[0])
losses = tf.losses.CategoricalCrossentropy(label_smoothing=0.01)

checkpoint_path = "save/" + train_name + "/log_train"

# model.layers[0].trainable = True
# for layer in model.layers[0].layers[:FINE_TUNE_START]:
#     layer.trainable = False
#
# model.compile(
#     loss=losses,
#     optimizer=tf.keras.optimizers.Adam(1e-5),
#     metrics=['accuracy'])
# # layer.trainable = False

model.layers[2].rate = 0
results = model.evaluate(val_generator)
print("test_model loss, test_model acc:", results)

'''Part3 Convert to TFLite'''
tf.saved_model.save(model, save_path)
converter = tf.lite.TFLiteConverter.from_saved_model(save_path)
tflite_model = converter.convert()
with open(save_path + '/' + train_name + '.tflite', 'wb') as f:
    f.write(tflite_model)
