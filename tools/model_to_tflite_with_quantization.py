import os
import pathlib

import numpy as np
import tensorflow as tf

from PIL import Image

from test_model.test_tflite_model import test_tflite_model


def representative_dataset():
    dataset_size = 100
    for i in range(dataset_size):
        img = Image.open("data/val_official/" + str(i) + ".jpg")
        img = img.resize((224, 224), Image.ANTIALIAS)
        img = tf.cast(img, tf.float32)

        img = tf.image.per_image_standardization(img)
        # data = imageio.imread("sample_images/" + str(i) + ".jpg")
        data = np.reshape(np.asarray(img), [1, 224, 224, 3])

        yield [data.astype(np.float32)]


# def representative_data_gen():
#     for input_value in tf.data.Dataset.from_tensor_slices(train_images).batch(1).take(100):
#         yield [input_value]


os.environ["CUDA_VISIBLE_DEVICES"] = "6"

train_name = "test_int8"
save_path = "save/" + train_name
converter = tf.lite.TFLiteConverter.from_saved_model(save_path)
quant_model = "dynamic"
if quant_model=="dynamic":
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

else:
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    # Ensure that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Set the input and output tensors to uint8 (APIs added in r2.3)
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

tflite_models_dir = pathlib.Path(save_path + '/')

tflite_model_quant = converter.convert()
tflite_model_quant_file = tflite_models_dir / str(train_name + '_int8.tflite')
tflite_model_quant_file.write_bytes(tflite_model_quant)

test_tflite_model(tflite_model_quant_file)
