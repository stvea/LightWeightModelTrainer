import os
import pathlib

import tensorflow as tf


def convert_from_save_model(model_save_path, type="normal", tflite_save_path=None):
    converter = tf.lite.TFLiteConverter.from_saved_model(model_save_path)

    if type == "dynamic":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    if tflite_save_path != None:
        with open(tflite_save_path, 'wb') as f:
            f.write(tflite_model)
    return tflite_model
