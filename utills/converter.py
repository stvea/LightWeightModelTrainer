import os
import pathlib
import numpy as np
import tensorflow as tf

from data_reader import img_preprocess


def representative_dataset():
    data_path = "/data2/competition/classification/represent_data/"
    for dirpath, dirnames, filenames in os.walk(data_path):
        for imgname in filenames:
            image = tf.io.read_file(os.path.join(data_path, imgname))
            image = tf.compat.v1.image.decode_jpeg(image)
            image = img_preprocess(image, 384, "per", False)
            image = np.array(image)
            print(os.path.join(data_path, imgname))

            image = np.reshape(image, (1, 384, 384, 3))
        yield [image.astype(np.float32)]


def convert_from_save_model(model_save_path, type="normal", tflite_save_path=None):
    converter = tf.lite.TFLiteConverter.from_saved_model(model_save_path)

    if type == "dynamic":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif type == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif type == "full_int":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset

    tflite_model = converter.convert()

    if tflite_save_path != None:
        with open(tflite_save_path, 'wb') as f:
            f.write(tflite_model)
    return tflite_save_path
