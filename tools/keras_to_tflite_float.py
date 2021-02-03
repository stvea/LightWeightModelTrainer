import os

import tensorflow as tf
from PIL import Image
import numpy as np

from model.shufflenet import ShufflenetV2

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

def representative_dataset():

    dataset_size = 10

    for i in range(dataset_size):
        img = Image.open("sample_images/" + str(i) + ".jpg")
        img = img.resize((224, 224), Image.ANTIALIAS)
        # data = imageio.imread("sample_images/" + str(i) + ".jpg")
        data = np.reshape(np.asarray(img), [1, 224, 224, 3])
        yield [data.astype(np.float32)]


def convert_model():

    model = ShufflenetV2(30, False)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer, loss=loss_object, metrics=['accuracy'])
    # model.build(input_shape=(1,224,224,3))

    latest = tf.train.latest_checkpoint("training_checkpoints/")
    model.load_weights(latest).expect_partial()
    # Load your pre-trained model
    # model.load_weights("path/to/your/saved/model")

    # Export your model to the TFLite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Be very careful here:
    # "experimental_new_converter" is enabled by default in TensorFlow 2.2+. However, using the new MLIR TFLite
    # converter might result in corrupted / incorrect TFLite models for some particular architectures. Therefore, the
    # best option is to perform the conversion using both the new and old converter and check the results in each case:
    converter.experimental_new_converter = True
    converter.experimental_new_quantizer = True

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()
    open("mobilenet_2_3_1st.tflite", "wb").write(tflite_model)

    # -----------------------------------------------------------------------------
    # That's it! Your model is now saved as model.tflite file
    # You can now try to run it using the PRO mode of the AI Benchmark application:
    # https://play.google.com/store/apps/details?id=org.benchmark.demo
    # More details can be found here (RUNTIME VALIDATION):
    # https://ai-benchmark.com/workshops/mai/2021/#runtime
    # -----------------------------------------------------------------------------


convert_model()